"""Reusable direct-subtree validation receipts for chaser relative frames.

The ordinary relative-frame source handle deliberately performs a complete
archive/consolidated deep audit.  That is the correct publication and audit
boundary, but it is unnecessarily expensive for every downstream consumer.
This module seals one completed immutable child after a direct-subtree audit
and lets bounded consumers validate the small child metadata generation and
rehash only the arrays they actually read.

The receipt is never a selector or production authority.  It is bound to one
absolute archive, one exact immutable run path, the complete declared array
hash table, the completion state, and every direct Zarr metadata file below
that run.  It intentionally does not parse the archive-root consolidated
metadata during reuse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.analysis_workflows.materializers.chaser_relative_frame import (
    MANIFEST_ATTR,
    MANIFEST_DIGEST_ATTR,
    PARENT_PATH,
)
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_io import open_zarr_root

RECEIPT_SCHEMA_ID = "palette.analysis.chaser_relative_frame.reusable_validation_receipt"
RECEIPT_SCHEMA_VERSION = 1
RECEIPT_STATUS = "complete_selector_ineligible_direct_subtree_audit"
VERIFICATION_MODE = "receipt_bound_targeted_array_rehash_v1"
VALIDATION_POLICY = {
    "initial_validation": ("streaming_all_declared_arrays_direct_content_sha256_v1"),
    "reuse_metadata_validation": "all_direct_subtree_zarr_json_sha256",
    "reuse_array_validation": "consumer_required_arrays_content_sha256",
    "archive_root_consolidated_metadata_reparse": False,
    "immutable_child_required": True,
}
STREAMING_HASH_BLOCK_BYTES = 32 * 1024 * 1024

OCCUPANCY_FRAME_ARRAY_NAMES = (
    "acquisition_frame_id",
    "timestamp_ns",
    "timestamp_valid",
    "selection_member",
    "fish_position_xy_px",
    "fish_position_valid",
)

DETAILED_PLOT_BASE_ARRAY_NAMES = (
    "acquisition_frame_id",
    "timestamp_ns",
    "timestamp_valid",
    "selection_member",
    "fish_position_xy_px",
    "fish_position_valid",
    "chaser_identity_code",
    "chaser_occurrence_member",
    "chaser_behavior_role_code",
    "chaser_position_xy_px",
    "chaser_position_valid",
    "relative_distance_physical",
    "relative_physical_valid",
)

_RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_SELECTOR_NAMES = {
    "latest",
    "latest_complete",
    "latest_provider",
    "latest_any",
    "selected",
    "selected_run",
    "active",
    "active_run",
    "current",
    "current_run",
    "default",
    "default_run",
}


class ChaserRelativeFrameValidationReceiptError(ValueError):
    """Raised when a reusable receipt or its exact child is invalid."""


def _fail(message: str) -> None:
    raise ChaserRelativeFrameValidationReceiptError(message)


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        _fail(f"{field} must be one non-empty exact string.")
    return value


def _digest(value: object, *, field: str) -> str:
    result = _text(value, field=field)
    if _SHA256_RE.fullmatch(result) is None:
        _fail(f"{field} must be one lowercase SHA-256 digest.")
    return result


def _git_commit(value: object, *, field: str) -> str:
    result = _text(value, field=field)
    if _GIT_COMMIT_RE.fullmatch(result) is None:
        _fail(f"{field} must be one full lowercase Git commit object ID.")
    return result


def _run_name(value: object) -> str:
    name = _text(value, field="run_name")
    if (
        _RUN_NAME_RE.fullmatch(name) is None
        or name.startswith(".")
        or name in _SELECTOR_NAMES
        or "/" in name
        or "\\" in name
    ):
        _fail("run_name must name one exact non-selector child.")
    return name


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ChaserRelativeFrameValidationReceiptError(
            f"Unable to read strict JSON object {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        _fail(f"JSON document is not one object: {path}")
    return value


def _metadata_inventory(run_directory: Path) -> dict[str, Any]:
    records = []
    for path in sorted(run_directory.rglob("zarr.json")):
        if not path.is_file():
            continue
        relative = path.relative_to(run_directory).as_posix()
        records.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    if not records or not any(record["path"] == "zarr.json" for record in records):
        _fail("Relative-frame child has no direct root zarr.json metadata.")
    return {
        "policy_id": "all_direct_subtree_zarr_json_files_sha256_v1",
        "file_count": len(records),
        "files": records,
        "inventory_sha256": canonical_json_sha256(records),
    }


def _streaming_array_values_sha256(
    node: Any,
    *,
    expected_dtype: str,
    expected_shape: Sequence[int],
) -> str:
    """Hash one Zarr array in bounded row-major slabs using Palette grammar."""

    dtype = np.dtype(node.dtype)
    shape = tuple(int(value) for value in node.shape)
    if dtype.str != expected_dtype or shape != tuple(expected_shape):
        _fail("Declared relative-frame array shape or dtype changed.")
    if dtype.hasobject or dtype.kind in {"U", "S"}:
        _fail("Declared relative-frame array dtype is not deterministic numeric data.")
    header = {
        "canonicalization": "numpy_dtype_shape_c_order_bytes_v1",
        "dtype": np.lib.format.dtype_to_descr(dtype),
        "shape": list(shape),
    }
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            header,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    digest.update(b"\x00")
    if not shape:
        block = np.ascontiguousarray(np.asarray(node[...]))
        digest.update(block.tobytes(order="C"))
        return digest.hexdigest()
    trailing_elements = int(np.prod(shape[1:], dtype=np.int64)) if shape[1:] else 1
    row_bytes = max(1, trailing_elements * dtype.itemsize)
    rows_per_block = max(1, STREAMING_HASH_BLOCK_BYTES // row_bytes)
    for start in range(0, shape[0], rows_per_block):
        stop = min(shape[0], start + rows_per_block)
        block = np.ascontiguousarray(np.asarray(node[start:stop, ...]))
        expected_block_shape = (stop - start, *shape[1:])
        if block.dtype.str != expected_dtype or block.shape != expected_block_shape:
            _fail("Relative-frame array changed during its streaming content audit.")
        digest.update(block.tobytes(order="C"))
        del block
    return digest.hexdigest()


def _streaming_direct_subtree_audit(
    run_directory: Path,
    *,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
) -> dict[str, Any]:
    """Rehash every declared array without retaining the complete run in RAM."""

    declarations = manifest.get("array_declarations")
    if not isinstance(declarations, list) or not declarations:
        _fail("Relative-frame manifest lacks array declarations.")
    declared_by_path: dict[str, Mapping[str, Any]] = {}
    for declaration in declarations:
        if not isinstance(declaration, Mapping):
            _fail("Relative-frame array declaration is not one object.")
        path = declaration.get("path")
        if type(path) is not str or path in declared_by_path or "/" not in path:
            _fail("Relative-frame array declaration paths are invalid or duplicated.")
        prefix, _name = path.split("/", 1)
        if prefix not in {"base", "body"}:
            _fail("Relative-frame array declaration leaves base/body.")
        declared_by_path[path] = declaration
    run = open_zarr_root(run_directory, mode="r", use_consolidated=False)
    actual_paths: set[str] = set()
    expected_prefixes = {path.split("/", 1)[0] for path in declared_by_path}
    actual_prefixes = set(str(name) for name in run.group_keys())
    if actual_prefixes != expected_prefixes:
        _fail("Relative-frame direct child groups differ from its declarations.")
    for prefix in sorted(expected_prefixes):
        group = run[prefix]
        if tuple(group.group_keys()):
            _fail("Relative-frame array group contains an unexpected nested group.")
        actual_paths.update(f"{prefix}/{name}" for name in group.array_keys())
    if actual_paths != set(declared_by_path):
        _fail("Relative-frame direct child arrays differ from its declarations.")
    content_bindings: dict[str, str] = {}
    for path, declaration in declared_by_path.items():
        expected_digest = _digest(
            declaration.get("content_sha256"),
            field=f"array_declarations[{path}].content_sha256",
        )
        observed = _streaming_array_values_sha256(
            run[path],
            expected_dtype=_text(
                declaration.get("dtype"),
                field=f"array_declarations[{path}].dtype",
            ),
            expected_shape=declaration.get("shape", ()),
        )
        if observed != expected_digest:
            _fail(f"Relative-frame array {path!r} content digest changed.")
        content_bindings[path] = observed
    return {
        "valid": True,
        "errors": [],
        "manifest_sha256": manifest_sha256,
        "array_count": len(declarations),
        "policy_id": "streaming_all_declared_arrays_direct_content_sha256_v1",
        "maximum_requested_block_bytes": STREAMING_HASH_BLOCK_BYTES,
        "content_bindings_sha256": canonical_json_sha256(content_bindings),
    }


def _current_run_document(
    analysis_zarr: str | Path,
    *,
    run_name: str,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    archive = Path(analysis_zarr).expanduser().resolve()
    if not archive.is_dir():
        raise FileNotFoundError(f"Analysis Zarr does not exist: {archive}")
    name = _run_name(run_name)
    run_directory = archive / PARENT_PATH / name
    metadata_path = run_directory / "zarr.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Relative-frame child does not exist: {run_directory}")
    document = _read_json_object(metadata_path)
    if document.get("node_type") != "group" or not isinstance(
        document.get("attributes"), dict
    ):
        _fail("Relative-frame child zarr.json is not one direct Zarr group.")
    return archive, run_directory, document, document["attributes"]


def build_chaser_relative_frame_validation_receipt(
    analysis_zarr: str | Path,
    *,
    run_name: str,
    palette_commit: str,
    expected_recording_id: str | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Deep-audit one immutable child directly and return a reusable receipt."""

    commit = _git_commit(palette_commit, field="palette_commit")
    archive, run_directory, _document, attrs = _current_run_document(
        analysis_zarr, run_name=run_name
    )
    name = _run_name(run_name)
    exact_run_path = f"{PARENT_PATH}/{name}"
    manifest = attrs.get(MANIFEST_ATTR)
    if not isinstance(manifest, dict):
        _fail("Relative-frame child lacks its exact run manifest.")
    manifest_digest = canonical_json_sha256(manifest)
    if attrs.get(MANIFEST_DIGEST_ATTR) != manifest_digest:
        _fail("Relative-frame child manifest digest is stale.")
    if (
        expected_manifest_sha256 is not None
        and _digest(expected_manifest_sha256, field="expected_manifest_sha256")
        != manifest_digest
    ):
        _fail("Relative-frame child differs from the expected manifest digest.")
    recording_id = _text(manifest.get("recording_id"), field="recording_id")
    if (
        expected_recording_id is not None
        and _text(expected_recording_id, field="expected_recording_id") != recording_id
    ):
        _fail("Relative-frame child belongs to another recording.")
    if manifest.get("run_path") not in (None, exact_run_path):
        _fail("Relative-frame manifest names another run path.")
    if (
        attrs.get("run_path") != exact_run_path
        or attrs.get("palette_run_name") != name
        or attrs.get("palette_run_completion_status") != "complete"
        or attrs.get("palette_run_completion_contract")
        != "palette.zarr_run_completion.v1"
        or attrs.get("selector_eligible") is not False
        or attrs.get("stage_selector_eligible") is not False
        or attrs.get("selection") != "none"
    ):
        _fail("Relative-frame child is not one completed selector-ineligible run.")
    direct_validation = _streaming_direct_subtree_audit(
        run_directory,
        manifest=manifest,
        manifest_sha256=manifest_digest,
    )
    if direct_validation.get("valid") is not True:
        _fail(f"Direct relative-frame deep audit failed: {direct_validation}")
    declarations = manifest.get("array_declarations")
    dimensions = manifest.get("dimensions")
    if not isinstance(declarations, list) or not isinstance(dimensions, dict):
        _fail("Relative-frame manifest lacks declarations or dimensions.")
    metadata = _metadata_inventory(run_directory)
    provenance = attrs.get("run_provenance")
    if not isinstance(provenance, dict):
        _fail("Relative-frame child lacks sealed run provenance.")
    body = {
        "schema_id": RECEIPT_SCHEMA_ID,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": RECEIPT_STATUS,
        "analysis_zarr": str(archive),
        "recording_id": recording_id,
        "run_name": name,
        "run_path": exact_run_path,
        "run_manifest": manifest,
        "manifest_sha256": manifest_digest,
        "payload_digest": _digest(
            manifest.get("payload_digest"), field="manifest.payload_digest"
        ),
        "dimensions": dimensions,
        "array_declarations": declarations,
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
        "direct_subtree_validation": direct_validation,
        "direct_metadata_inventory": metadata,
        "software_authority": {"repository": "palette", "commit": commit},
        "validation_policy": VALIDATION_POLICY,
        "safety": {
            "selector_eligible": False,
            "production_authority": False,
            "registry_update": False,
            "selector_activation": False,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return {**body, "record_sha256": canonical_json_sha256(body)}


def validate_chaser_relative_frame_validation_receipt(
    receipt: object,
    *,
    expected_analysis_zarr: str | Path | None = None,
    expected_recording_id: str | None = None,
    expected_run_name: str | None = None,
    expected_manifest_sha256: str | None = None,
    validate_current_metadata: bool = True,
) -> Mapping[str, Any]:
    """Validate a receipt envelope and, by default, its current child metadata."""

    if not isinstance(receipt, Mapping):
        _fail("Relative-frame validation receipt must be one object.")
    body = _plain(receipt)
    record_digest = body.pop("record_sha256", None)
    if _digest(record_digest, field="record_sha256") != canonical_json_sha256(body):
        _fail("Relative-frame validation receipt digest is stale.")
    required = {
        "schema_id",
        "schema_version",
        "status",
        "analysis_zarr",
        "recording_id",
        "run_name",
        "run_path",
        "run_manifest",
        "manifest_sha256",
        "payload_digest",
        "dimensions",
        "array_declarations",
        "run_provenance_sha256",
        "completion_authority",
        "direct_subtree_validation",
        "direct_metadata_inventory",
        "software_authority",
        "validation_policy",
        "safety",
        "created_at_utc",
    }
    if set(body) != required:
        _fail("Relative-frame validation receipt fields are missing or unexpected.")
    if (
        body["schema_id"] != RECEIPT_SCHEMA_ID
        or body["schema_version"] != RECEIPT_SCHEMA_VERSION
        or body["status"] != RECEIPT_STATUS
        or body["safety"]
        != {
            "selector_eligible": False,
            "production_authority": False,
            "registry_update": False,
            "selector_activation": False,
        }
    ):
        _fail("Relative-frame validation receipt identity or safety is invalid.")
    name = _run_name(body["run_name"])
    if body["run_path"] != f"{PARENT_PATH}/{name}":
        _fail("Relative-frame validation receipt run path is invalid.")
    archive = Path(_text(body["analysis_zarr"], field="analysis_zarr")).resolve()
    if not archive.is_absolute():
        _fail("Relative-frame validation receipt archive path must be absolute.")
    recording_id = _text(body["recording_id"], field="recording_id")
    manifest = body["run_manifest"]
    if not isinstance(manifest, dict):
        _fail("Receipt run_manifest is not one object.")
    manifest_digest = _digest(body["manifest_sha256"], field="manifest_sha256")
    if manifest_digest != canonical_json_sha256(manifest):
        _fail("Receipt run manifest digest is stale.")
    payload_digest = _digest(body["payload_digest"], field="payload_digest")
    if (
        manifest.get("recording_id") != recording_id
        or manifest.get("run_path") not in (None, body["run_path"])
        or manifest.get("payload_digest") != payload_digest
    ):
        _fail("Receipt run manifest identity or payload binding is invalid.")
    _digest(body["run_provenance_sha256"], field="run_provenance_sha256")
    software = body["software_authority"]
    if (
        not isinstance(software, dict)
        or set(software) != {"repository", "commit"}
        or software["repository"] != "palette"
    ):
        _fail("Receipt software authority is invalid.")
    _git_commit(software["commit"], field="software_authority.commit")
    if body["validation_policy"] != VALIDATION_POLICY:
        _fail("Receipt validation policy is invalid.")
    if (
        expected_analysis_zarr is not None
        and archive != Path(expected_analysis_zarr).expanduser().resolve()
    ):
        _fail("Receipt names another analysis archive.")
    if expected_recording_id is not None and recording_id != _text(
        expected_recording_id, field="expected_recording_id"
    ):
        _fail("Receipt names another recording.")
    if expected_run_name is not None and name != _run_name(expected_run_name):
        _fail("Receipt names another relative-frame run.")
    if expected_manifest_sha256 is not None and manifest_digest != _digest(
        expected_manifest_sha256, field="expected_manifest_sha256"
    ):
        _fail("Receipt names another relative-frame manifest.")
    if body["array_declarations"] != manifest.get("array_declarations"):
        _fail("Receipt array declarations differ from its run manifest.")
    if body["dimensions"] != manifest.get("dimensions"):
        _fail("Receipt dimensions differ from its run manifest.")
    declarations = body["array_declarations"]
    if not isinstance(declarations, list) or not declarations:
        _fail("Receipt array declarations must be one non-empty list.")
    declared_paths: list[str] = []
    for index, declaration in enumerate(declarations):
        if not isinstance(declaration, dict) or set(declaration) != {
            "path",
            "dtype",
            "shape",
            "content_sha256",
        }:
            _fail(f"Receipt array declaration {index} is invalid.")
        path_value = _text(
            declaration["path"], field=f"array_declarations[{index}].path"
        )
        if path_value.startswith("/") or ".." in Path(path_value).parts:
            _fail("Receipt array declaration path leaves the exact child.")
        _text(declaration["dtype"], field=f"array_declarations[{index}].dtype")
        shape = declaration["shape"]
        if (
            not isinstance(shape, list)
            or not shape
            or any(type(value) is not int or value < 0 for value in shape)
        ):
            _fail(f"Receipt array declaration {index} shape is invalid.")
        _digest(
            declaration["content_sha256"],
            field=f"array_declarations[{index}].content_sha256",
        )
        declared_paths.append(path_value)
    if len(declared_paths) != len(set(declared_paths)):
        _fail("Receipt array declarations contain duplicate paths.")
    completion = body["completion_authority"]
    if (
        not isinstance(completion, dict)
        or set(completion)
        != {"contract", "status", "completed_at_utc", "publication_owner_uuid"}
        or completion["contract"] != "palette.zarr_run_completion.v1"
        or completion["status"] != "complete"
    ):
        _fail("Receipt completion authority is invalid.")
    direct_validation = body["direct_subtree_validation"]
    if (
        not isinstance(direct_validation, dict)
        or direct_validation.get("valid") is not True
        or direct_validation.get("manifest_sha256") != manifest_digest
        or int(direct_validation.get("array_count", -1)) != len(declarations)
    ):
        _fail("Receipt direct-subtree validation evidence is invalid.")
    inventory = body["direct_metadata_inventory"]
    if (
        not isinstance(inventory, dict)
        or set(inventory) != {"policy_id", "file_count", "files", "inventory_sha256"}
        or inventory["policy_id"] != "all_direct_subtree_zarr_json_files_sha256_v1"
        or not isinstance(inventory["files"], list)
        or int(inventory["file_count"]) != len(inventory["files"])
        or canonical_json_sha256(inventory["files"])
        != _digest(inventory["inventory_sha256"], field="inventory_sha256")
    ):
        _fail("Receipt direct metadata inventory is invalid.")
    metadata_paths = []
    for index, record in enumerate(inventory["files"]):
        if not isinstance(record, dict) or set(record) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            _fail(f"Receipt metadata inventory record {index} is invalid.")
        metadata_path = _text(
            record["path"], field=f"direct_metadata_inventory.files[{index}].path"
        )
        if (
            Path(metadata_path).name != "zarr.json"
            or metadata_path.startswith("/")
            or ".." in Path(metadata_path).parts
            or type(record["size_bytes"]) is not int
            or record["size_bytes"] < 0
        ):
            _fail(f"Receipt metadata inventory record {index} is unsafe.")
        _digest(
            record["sha256"],
            field=f"direct_metadata_inventory.files[{index}].sha256",
        )
        metadata_paths.append(metadata_path)
    if "zarr.json" not in metadata_paths or len(metadata_paths) != len(
        set(metadata_paths)
    ):
        _fail("Receipt metadata inventory root or uniqueness is invalid.")
    if validate_current_metadata:
        _, run_directory, _document, attrs = _current_run_document(
            archive, run_name=name
        )
        if (
            attrs.get(MANIFEST_ATTR) != manifest
            or attrs.get(MANIFEST_DIGEST_ATTR) != manifest_digest
            or attrs.get("run_provenance") is None
            or canonical_json_sha256(attrs["run_provenance"])
            != body["run_provenance_sha256"]
            or attrs.get("palette_run_completion_status") != "complete"
            or attrs.get("atomic_publication_owner_uuid")
            != completion["publication_owner_uuid"]
        ):
            _fail("Receipt-bound relative-frame child metadata changed.")
        if _metadata_inventory(run_directory) != body["direct_metadata_inventory"]:
            _fail("Receipt-bound direct subtree metadata generation changed.")
    return MappingProxyType({**body, "record_sha256": record_digest})


def read_chaser_relative_frame_validation_receipt(
    path: str | Path,
    **expected: Any,
) -> Mapping[str, Any]:
    receipt_path = Path(path).expanduser().resolve()
    return validate_chaser_relative_frame_validation_receipt(
        _read_json_object(receipt_path), **expected
    )


def ensure_chaser_relative_frame_validation_receipt(
    analysis_zarr: str | Path,
    *,
    run_name: str,
    palette_commit: str,
    output_json: str | Path,
    expected_recording_id: str | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Create one receipt, or validate and exactly reuse an existing receipt."""

    output = Path(output_json).expanduser().resolve()
    if output.exists():
        validated = read_chaser_relative_frame_validation_receipt(
            output,
            expected_analysis_zarr=analysis_zarr,
            expected_recording_id=expected_recording_id,
            expected_run_name=run_name,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        if validated["software_authority"]["commit"] != _git_commit(
            palette_commit, field="palette_commit"
        ):
            _fail("Existing validation receipt was created by another commit.")
        return {
            **_plain(validated),
            "receipt_path": str(output),
            "mode": "reused_exact",
        }
    receipt = build_chaser_relative_frame_validation_receipt(
        analysis_zarr,
        run_name=run_name,
        palette_commit=palette_commit,
        expected_recording_id=expected_recording_id,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    write_json_atomic(output, receipt, overwrite=False)
    return {**receipt, "receipt_path": str(output), "mode": "created"}


@dataclass(frozen=True, slots=True)
class ChaserRelativeFrameTargetedSourceHandle:
    """Receipt-bound read-only relative-frame arrays for bounded consumers."""

    analysis_zarr_path: Path
    run_path: str
    run_name: str
    recording_id: str
    manifest_sha256: str
    payload_digest: str
    n_frames: int
    n_chasers: int
    n_rows: int
    run_manifest: Mapping[str, Any] = field(repr=False, compare=False)
    source_authorities: Mapping[str, Any] = field(repr=False, compare=False)
    base_arrays: Mapping[str, np.ndarray] = field(repr=False, compare=False)
    body_arrays: Mapping[str, np.ndarray] = field(repr=False, compare=False)
    frame_arrays: Mapping[str, np.ndarray] = field(repr=False, compare=False)
    receipt: Mapping[str, Any] = field(repr=False, compare=False)
    receipt_path: Path = field(repr=False, compare=False)
    receipt_digest: str
    verification_digest: str
    verification_mode: str = VERIFICATION_MODE

    @property
    def manifest(self) -> Mapping[str, Any]:
        return self.run_manifest

    def base_array(self, name: str) -> np.ndarray:
        try:
            return self.base_arrays[name]
        except KeyError as exc:
            raise KeyError(f"Targeted relative-frame handle lacks {name!r}.") from exc

    def base_frame_chaser(self, name: str) -> np.ndarray:
        values = self.base_array(name)
        return values.reshape((self.n_frames, self.n_chasers) + values.shape[1:])

    def body_array(self, name: str) -> np.ndarray:
        try:
            return self.body_arrays[name]
        except KeyError as exc:
            raise KeyError(
                f"Targeted relative-frame handle lacks body array {name!r}."
            ) from exc

    def body_frame_chaser(self, name: str) -> np.ndarray:
        values = self.body_array(name)
        return values.reshape((self.n_frames, self.n_chasers) + values.shape[1:])

    def frame_array(self, name: str) -> np.ndarray:
        try:
            return self.frame_arrays[name]
        except KeyError as exc:
            raise KeyError(f"Targeted relative-frame handle lacks {name!r}.") from exc

    def assert_current(self) -> None:
        read_chaser_relative_frame_validation_receipt(
            self.receipt_path,
            expected_analysis_zarr=self.analysis_zarr_path,
            expected_recording_id=self.recording_id,
            expected_run_name=self.run_name,
            expected_manifest_sha256=self.manifest_sha256,
        )


def load_chaser_relative_frame_targeted_source_handle(
    receipt_path: str | Path,
    *,
    required_base_arrays: Sequence[str] = OCCUPANCY_FRAME_ARRAY_NAMES,
    required_body_arrays: Sequence[str] = (),
    collapsed_frame_arrays: Sequence[str] = OCCUPANCY_FRAME_ARRAY_NAMES,
    expected_analysis_zarr: str | Path | None = None,
    expected_recording_id: str | None = None,
    expected_run_name: str | None = None,
) -> ChaserRelativeFrameTargetedSourceHandle:
    """Load and content-rehash only the explicitly requested arrays."""

    path = Path(receipt_path).expanduser().resolve()
    receipt = read_chaser_relative_frame_validation_receipt(
        path,
        expected_analysis_zarr=expected_analysis_zarr,
        expected_recording_id=expected_recording_id,
        expected_run_name=expected_run_name,
    )
    names = tuple(required_base_arrays)
    if (
        not names
        or len(names) != len(set(names))
        or any(type(name) is not str or not name for name in names)
    ):
        _fail("required_base_arrays must be unique non-empty strings.")
    body_names = tuple(required_body_arrays)
    if len(body_names) != len(set(body_names)) or any(
        type(name) is not str or not name for name in body_names
    ):
        _fail("required_body_arrays must be unique non-empty strings.")
    collapsed = tuple(collapsed_frame_arrays)
    if len(collapsed) != len(set(collapsed)) or not set(collapsed).issubset(names):
        _fail("collapsed_frame_arrays must be a unique subset of required arrays.")
    declarations = {item["path"]: item for item in receipt["array_declarations"]}
    missing = [name for name in names if f"base/{name}" not in declarations]
    if missing:
        _fail(f"Receipt omits required targeted arrays: {missing!r}")
    missing_body = [name for name in body_names if f"body/{name}" not in declarations]
    if missing_body:
        _fail(f"Receipt omits required targeted body arrays: {missing_body!r}")
    dimensions = receipt["dimensions"]
    n_frames = int(dimensions["n_frames"])
    n_chasers = int(dimensions["n_chasers"])
    n_rows = int(dimensions["n_rows"])
    if n_frames <= 0 or n_chasers <= 0 or n_rows != n_frames * n_chasers:
        _fail("Receipt frame/chaser dimensions are invalid.")
    run_directory = Path(receipt["analysis_zarr"]) / PARENT_PATH / receipt["run_name"]
    run = open_zarr_root(run_directory, mode="r", use_consolidated=False)
    base = run["base"]
    base_arrays: dict[str, np.ndarray] = {}
    body_arrays: dict[str, np.ndarray] = {}
    frame_arrays: dict[str, np.ndarray] = {}
    content_bindings: dict[str, str] = {}
    for name in names:
        declaration = declarations[f"base/{name}"]
        values = np.asarray(base[name][...])
        if (
            values.dtype.str != declaration["dtype"]
            or list(values.shape) != declaration["shape"]
        ):
            _fail(f"Targeted array base/{name} shape or dtype changed.")
        digest = array_values_sha256(values)
        if digest != declaration["content_sha256"]:
            _fail(f"Targeted array base/{name} content digest changed.")
        base_values = np.array(values, copy=True, order="C")
        base_values.setflags(write=False)
        base_arrays[name] = base_values
        if name in collapsed:
            reshaped = values.reshape((n_frames, n_chasers) + values.shape[1:])
            reference = reshaped[:, :1, ...]
            if values.dtype.kind == "f":
                repeated = np.array_equal(
                    reshaped,
                    np.broadcast_to(reference, reshaped.shape),
                    equal_nan=True,
                )
            else:
                repeated = np.array_equal(
                    reshaped, np.broadcast_to(reference, reshaped.shape)
                )
            if not repeated:
                _fail(f"Targeted frame array base/{name} differs across chasers.")
            frame_values = np.array(reshaped[:, 0, ...], copy=True, order="C")
            frame_values.setflags(write=False)
            frame_arrays[name] = frame_values
        content_bindings[f"base/{name}"] = digest
    if body_names:
        body = run["body"]
        for name in body_names:
            declaration = declarations[f"body/{name}"]
            values = np.asarray(body[name][...])
            if (
                values.dtype.str != declaration["dtype"]
                or list(values.shape) != declaration["shape"]
                or values.shape[0] != n_rows
            ):
                _fail(f"Targeted array body/{name} shape or dtype changed.")
            digest = array_values_sha256(values)
            if digest != declaration["content_sha256"]:
                _fail(f"Targeted array body/{name} content digest changed.")
            body_values = np.array(values, copy=True, order="C")
            body_values.setflags(write=False)
            body_arrays[name] = body_values
            content_bindings[f"body/{name}"] = digest
    manifest = receipt["run_manifest"]
    authorities = manifest.get("source_authorities")
    if not isinstance(authorities, Mapping):
        _fail("Receipt-bound manifest lacks source authorities.")
    verification_document = {
        "receipt_sha256": receipt["record_sha256"],
        "manifest_sha256": receipt["manifest_sha256"],
        "required_array_content_sha256": content_bindings,
        "verification_mode": VERIFICATION_MODE,
    }
    return ChaserRelativeFrameTargetedSourceHandle(
        analysis_zarr_path=Path(receipt["analysis_zarr"]),
        run_path=receipt["run_path"],
        run_name=receipt["run_name"],
        recording_id=receipt["recording_id"],
        manifest_sha256=receipt["manifest_sha256"],
        payload_digest=receipt["payload_digest"],
        n_frames=n_frames,
        n_chasers=n_chasers,
        n_rows=n_rows,
        run_manifest=_freeze(manifest),
        source_authorities=_freeze(authorities),
        base_arrays=MappingProxyType(base_arrays),
        body_arrays=MappingProxyType(body_arrays),
        frame_arrays=MappingProxyType(frame_arrays),
        receipt=_freeze(receipt),
        receipt_path=path,
        receipt_digest=receipt["record_sha256"],
        verification_digest=canonical_json_sha256(verification_document),
    )


__all__ = [
    "DETAILED_PLOT_BASE_ARRAY_NAMES",
    "OCCUPANCY_FRAME_ARRAY_NAMES",
    "RECEIPT_SCHEMA_ID",
    "RECEIPT_SCHEMA_VERSION",
    "VERIFICATION_MODE",
    "ChaserRelativeFrameTargetedSourceHandle",
    "ChaserRelativeFrameValidationReceiptError",
    "build_chaser_relative_frame_validation_receipt",
    "ensure_chaser_relative_frame_validation_receipt",
    "load_chaser_relative_frame_targeted_source_handle",
    "read_chaser_relative_frame_validation_receipt",
    "validate_chaser_relative_frame_validation_receipt",
]
