"""Versioned integrity and validation receipts for immutable Zarr run payloads.

The receipt deliberately separates three facts:

* decoded shard leaves prove which logical values were copied and read back;
* physical payload leaves prove that the installed immutable data objects have
  not changed; and
* a validation receipt binds one versioned scientific validator to the exact
  integrity root and scientific-manifest digest it accepted.

Zarr ``zarr.json`` files are excluded from the physical payload root.  A
publisher may therefore add coordinate, completion, or selector metadata while
the receipt continues to bind the immutable array payload.  Metadata bindings
must be validated separately by the owning stage contract.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

PAYLOAD_INTEGRITY_RECEIPT_SCHEMA_ID = "palette.zarr_payload_integrity_receipt"
PAYLOAD_INTEGRITY_RECEIPT_SCHEMA_VERSION = 1
PAYLOAD_VALIDATION_RECEIPT_SCHEMA_ID = "palette.zarr_payload_validation_receipt"
PAYLOAD_VALIDATION_RECEIPT_SCHEMA_VERSION = 1
DECODED_PAYLOAD_CANONICALIZATION = "numpy_declared_dtype_shape_c_order_bytes_v1"
PHYSICAL_PAYLOAD_CANONICALIZATION = "relative_path_size_sha256_v1"
IMMUTABLE_METADATA_CANONICALIZATION = (
    "zarr_json_without_attributes_relative_path_sha256_v1"
)
MERKLE_COMPOSITION = "canonical_json_sha256_sorted_children_v1"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: Any, *, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{label} must be one lowercase SHA-256 digest.")
    return str(value)


def _canonical_copy(value: Any, *, label: str) -> Any:
    try:
        return json.loads(_canonical_json(value).decode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be strict canonical JSON: {exc}.") from exc


def _payload_files(root: Path) -> tuple[Path, ...]:
    return tuple(
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "zarr.json"
    )


def _hash_file(item: tuple[Path, Path]) -> dict[str, Any]:
    root, path = item
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": int(path.stat().st_size),
        "sha256": digest.hexdigest(),
    }


def _physical_payload_record(root: Path, *, workers: int) -> dict[str, Any]:
    if type(workers) is not int or workers <= 0:
        raise ValueError("Physical payload hash workers must be positive.")
    paths = _payload_files(root)
    if not paths:
        raise ValueError(f"Zarr run has no physical payload files: {root}")
    if workers == 1 or len(paths) == 1:
        leaves = [_hash_file((root, path)) for path in paths]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(paths))) as executor:
            leaves = list(executor.map(_hash_file, ((root, path) for path in paths)))
    leaves.sort(key=lambda value: str(value["relative_path"]))
    return {
        "canonicalization": PHYSICAL_PAYLOAD_CANONICALIZATION,
        "file_count": len(leaves),
        "total_bytes": sum(int(value["size_bytes"]) for value in leaves),
        "root_sha256": canonical_json_sha256(leaves),
        "files": leaves,
    }


def _immutable_metadata_record(root: Path) -> dict[str, Any]:
    leaves: list[dict[str, Any]] = []
    for path in sorted(root.rglob("zarr.json")):
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid Zarr metadata file {path}: {exc}.") from exc
        if not isinstance(metadata, dict):
            raise ValueError(f"Zarr metadata file is not an object: {path}")
        immutable = dict(metadata)
        attributes = immutable.pop("attributes", {})
        if not isinstance(attributes, dict):
            raise ValueError(f"Zarr metadata attributes are not an object: {path}")
        leaves.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "node_type": immutable.get("node_type"),
                "sha256": canonical_json_sha256(immutable),
            }
        )
    if not leaves:
        raise ValueError(f"Zarr run has no zarr.json metadata files: {root}")
    return {
        "canonicalization": IMMUTABLE_METADATA_CANONICALIZATION,
        "file_count": len(leaves),
        "root_sha256": canonical_json_sha256(leaves),
        "files": leaves,
    }


def _decoded_payload_record(copy_report: Mapping[str, Any]) -> dict[str, Any]:
    if (
        copy_report.get("schema_id") != "palette.zarr_sharded_run_copy.v1"
        or copy_report.get("status") != "complete"
        or copy_report.get("exact_decoded_validation") is not True
    ):
        raise ValueError(
            "Decoded payload receipt requires one exact completed sharded copy."
        )
    raw_plans = copy_report.get("arrays")
    raw_shards = copy_report.get("shards")
    raw_static = copy_report.get("static_arrays")
    if not isinstance(raw_plans, Sequence) or isinstance(raw_plans, (str, bytes)):
        raise ValueError("Sharded copy lacks its closed array plan inventory.")
    if not isinstance(raw_shards, Sequence) or isinstance(raw_shards, (str, bytes)):
        raise ValueError("Sharded copy lacks its decoded shard inventory.")
    if not isinstance(raw_static, Sequence) or isinstance(raw_static, (str, bytes)):
        raise ValueError("Sharded copy lacks its decoded static-array inventory.")

    plans: dict[str, dict[str, Any]] = {}
    for raw_plan in raw_plans:
        if not isinstance(raw_plan, Mapping):
            raise ValueError("Sharded copy array plan is malformed.")
        path = raw_plan.get("path")
        shape = raw_plan.get("shape")
        dtype = raw_plan.get("dtype")
        if (
            type(path) is not str
            or not path
            or path in plans
            or not isinstance(shape, Sequence)
            or isinstance(shape, (str, bytes))
            or any(type(value) is not int or value < 0 for value in shape)
            or type(dtype) is not str
            or not dtype
        ):
            raise ValueError("Sharded copy array plan identity is invalid.")
        plans[path] = {
            "path": path,
            "dtype": dtype,
            "shape": [int(value) for value in shape],
        }

    leaves_by_array: dict[str, list[dict[str, Any]]] = {path: [] for path in plans}
    for kind, raw_records in (("row_shard", raw_shards), ("whole_array", raw_static)):
        for raw_record in raw_records:
            if not isinstance(raw_record, Mapping):
                raise ValueError("Decoded copy leaf is malformed.")
            path = raw_record.get("path")
            if path not in plans:
                raise ValueError("Decoded copy leaf names an undeclared array.")
            decoded_bytes = raw_record.get("decoded_bytes")
            digest = raw_record.get("decoded_sha256")
            if type(decoded_bytes) is not int or decoded_bytes < 0:
                raise ValueError("Decoded copy leaf byte count is invalid.")
            _require_sha256(digest, label="decoded copy leaf digest")
            leaf: dict[str, Any] = {
                "kind": kind,
                "decoded_bytes": int(decoded_bytes),
                "decoded_sha256": str(digest),
            }
            if kind == "row_shard":
                start = raw_record.get("start_row")
                stop = raw_record.get("stop_row")
                if (
                    type(start) is not int
                    or type(stop) is not int
                    or not (0 <= start < stop <= plans[str(path)]["shape"][0])
                ):
                    raise ValueError("Decoded row-shard bounds are invalid.")
                leaf.update({"start_row": int(start), "stop_row": int(stop)})
            leaves_by_array[str(path)].append(leaf)

    arrays: list[dict[str, Any]] = []
    for path, plan in sorted(plans.items()):
        leaves = leaves_by_array[path]
        if not leaves:
            if plan["shape"] and plan["shape"][0] != 0:
                raise ValueError(f"Decoded copy has no payload leaves for {path!r}.")
            leaves = [
                {
                    "kind": "empty_array",
                    "decoded_bytes": 0,
                    "decoded_sha256": hashlib.sha256(b"").hexdigest(),
                }
            ]
        leaves.sort(
            key=lambda value: (
                str(value["kind"]),
                int(value.get("start_row", -1)),
                int(value.get("stop_row", -1)),
            )
        )
        row_leaves = [value for value in leaves if value["kind"] == "row_shard"]
        if row_leaves:
            cursor = 0
            for leaf in row_leaves:
                if leaf["start_row"] != cursor:
                    raise ValueError(
                        f"Decoded row-shard leaves have a gap or overlap for {path!r}."
                    )
                cursor = int(leaf["stop_row"])
            if cursor != plan["shape"][0] or len(row_leaves) != len(leaves):
                raise ValueError(
                    f"Decoded row-shard leaves do not exactly cover {path!r}."
                )
        elif len(leaves) != 1:
            raise ValueError(f"Static decoded array {path!r} has multiple leaves.")
        arrays.append(
            {
                **plan,
                "decoded_bytes": sum(int(value["decoded_bytes"]) for value in leaves),
                "leaf_count": len(leaves),
                "root_sha256": canonical_json_sha256(leaves),
                "leaves": leaves,
            }
        )
    return {
        "canonicalization": DECODED_PAYLOAD_CANONICALIZATION,
        "array_count": len(arrays),
        "decoded_bytes": sum(int(value["decoded_bytes"]) for value in arrays),
        "root_sha256": canonical_json_sha256(
            [
                {
                    "path": value["path"],
                    "dtype": value["dtype"],
                    "shape": value["shape"],
                    "root_sha256": value["root_sha256"],
                }
                for value in arrays
            ]
        ),
        "arrays": arrays,
    }


def decoded_payload_receipt_from_copy_report(
    copy_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the canonical decoded-payload receipt for an exact copy report.

    This intentionally excludes physical files and mutable Zarr attributes.  It
    lets a local scientific validator bind its result to the same decoded array
    identity that the final authoritative payload-integrity receipt will carry.
    """

    return _decoded_payload_record(copy_report)


def _validate_decoded_payload_record(value: Any) -> dict[str, Any]:
    decoded = _canonical_copy(value, label="decoded payload receipt")
    if not isinstance(decoded, dict) or set(decoded) != {
        "canonicalization",
        "array_count",
        "decoded_bytes",
        "root_sha256",
        "arrays",
    }:
        raise ValueError("Decoded payload receipt fields are not exact.")
    arrays = decoded.get("arrays")
    if (
        decoded.get("canonicalization") != DECODED_PAYLOAD_CANONICALIZATION
        or type(decoded.get("array_count")) is not int
        or decoded["array_count"] < 0
        or type(decoded.get("decoded_bytes")) is not int
        or decoded["decoded_bytes"] < 0
        or not isinstance(arrays, list)
        or decoded["array_count"] != len(arrays)
    ):
        raise ValueError("Decoded payload receipt inventory is invalid.")

    canonical_arrays: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for array in arrays:
        if not isinstance(array, dict) or set(array) != {
            "path",
            "dtype",
            "shape",
            "decoded_bytes",
            "leaf_count",
            "root_sha256",
            "leaves",
        }:
            raise ValueError("Decoded payload array receipt fields are not exact.")
        path = array.get("path")
        shape = array.get("shape")
        leaves = array.get("leaves")
        if (
            type(path) is not str
            or not path
            or path in seen_paths
            or type(array.get("dtype")) is not str
            or not array["dtype"]
            or not isinstance(shape, list)
            or any(type(item) is not int or item < 0 for item in shape)
            or type(array.get("decoded_bytes")) is not int
            or array["decoded_bytes"] < 0
            or type(array.get("leaf_count")) is not int
            or not isinstance(leaves, list)
            or array["leaf_count"] != len(leaves)
        ):
            raise ValueError("Decoded payload array receipt inventory is invalid.")
        seen_paths.add(path)
        row_cursor = 0
        row_leaf_count = 0
        decoded_bytes = 0
        for leaf in leaves:
            if not isinstance(leaf, dict):
                raise ValueError("Decoded payload leaf is malformed.")
            kind = leaf.get("kind")
            expected_fields = {"kind", "decoded_bytes", "decoded_sha256"}
            if kind == "row_shard":
                expected_fields.update({"start_row", "stop_row"})
            if (
                set(leaf) != expected_fields
                or kind not in {"row_shard", "whole_array", "empty_array"}
                or type(leaf.get("decoded_bytes")) is not int
                or leaf["decoded_bytes"] < 0
            ):
                raise ValueError("Decoded payload leaf fields are invalid.")
            _require_sha256(
                leaf.get("decoded_sha256"), label="decoded payload leaf digest"
            )
            decoded_bytes += int(leaf["decoded_bytes"])
            if kind == "row_shard":
                start = leaf.get("start_row")
                stop = leaf.get("stop_row")
                if (
                    type(start) is not int
                    or type(stop) is not int
                    or start != row_cursor
                    or not (start < stop)
                    or not shape
                    or stop > shape[0]
                ):
                    raise ValueError(
                        "Decoded payload row leaves have a gap, overlap, or invalid bound."
                    )
                row_cursor = stop
                row_leaf_count += 1
        if row_leaf_count:
            if row_leaf_count != len(leaves) or row_cursor != shape[0]:
                raise ValueError("Decoded payload row leaves do not exactly cover array.")
        elif len(leaves) != 1 or (
            leaves[0]["kind"] == "empty_array" and (not shape or shape[0] != 0)
        ):
            raise ValueError("Decoded static or empty array leaves are invalid.")
        if (
            array["decoded_bytes"] != decoded_bytes
            or array.get("root_sha256") != canonical_json_sha256(leaves)
        ):
            raise ValueError("Decoded payload array root or byte count is stale.")
        canonical_arrays.append(array)

    if canonical_arrays != sorted(canonical_arrays, key=lambda item: item["path"]):
        raise ValueError("Decoded payload arrays are not canonically ordered.")
    root_children = [
        {
            "path": array["path"],
            "dtype": array["dtype"],
            "shape": array["shape"],
            "root_sha256": array["root_sha256"],
        }
        for array in canonical_arrays
    ]
    if (
        decoded["decoded_bytes"]
        != sum(int(array["decoded_bytes"]) for array in canonical_arrays)
        or decoded.get("root_sha256") != canonical_json_sha256(root_children)
    ):
        raise ValueError("Decoded payload root or byte count is stale.")
    return decoded


def _validate_physical_payload_record(value: Any) -> dict[str, Any]:
    physical = _canonical_copy(value, label="physical payload receipt")
    if not isinstance(physical, dict) or set(physical) != {
        "canonicalization",
        "file_count",
        "total_bytes",
        "root_sha256",
        "files",
    }:
        raise ValueError("Physical payload receipt fields are not exact.")
    files = physical.get("files")
    if (
        physical.get("canonicalization") != PHYSICAL_PAYLOAD_CANONICALIZATION
        or type(physical.get("file_count")) is not int
        or physical["file_count"] < 1
        or type(physical.get("total_bytes")) is not int
        or physical["total_bytes"] < 0
        or not isinstance(files, list)
        or physical["file_count"] != len(files)
    ):
        raise ValueError("Physical payload receipt inventory is invalid.")
    seen_paths: set[str] = set()
    for file_record in files:
        if not isinstance(file_record, dict) or set(file_record) != {
            "relative_path",
            "size_bytes",
            "sha256",
        }:
            raise ValueError("Physical payload file receipt fields are not exact.")
        relative_path = file_record.get("relative_path")
        if (
            type(relative_path) is not str
            or not relative_path
            or relative_path.startswith("/")
            or ".." in Path(relative_path).parts
            or Path(relative_path).name == "zarr.json"
            or relative_path in seen_paths
            or type(file_record.get("size_bytes")) is not int
            or file_record["size_bytes"] < 0
        ):
            raise ValueError("Physical payload file receipt identity is invalid.")
        _require_sha256(file_record.get("sha256"), label="physical payload digest")
        seen_paths.add(relative_path)
    if files != sorted(files, key=lambda item: item["relative_path"]):
        raise ValueError("Physical payload files are not canonically ordered.")
    if (
        physical["total_bytes"]
        != sum(int(file_record["size_bytes"]) for file_record in files)
        or physical.get("root_sha256") != canonical_json_sha256(files)
    ):
        raise ValueError("Physical payload root or byte count is stale.")
    return physical


def _validate_immutable_metadata_record(value: Any) -> dict[str, Any]:
    metadata = _canonical_copy(value, label="immutable Zarr metadata receipt")
    if not isinstance(metadata, dict) or set(metadata) != {
        "canonicalization",
        "file_count",
        "root_sha256",
        "files",
    }:
        raise ValueError("Immutable Zarr metadata receipt fields are not exact.")
    files = metadata.get("files")
    if (
        metadata.get("canonicalization") != IMMUTABLE_METADATA_CANONICALIZATION
        or type(metadata.get("file_count")) is not int
        or metadata["file_count"] < 1
        or not isinstance(files, list)
        or metadata["file_count"] != len(files)
    ):
        raise ValueError("Immutable Zarr metadata receipt inventory is invalid.")
    seen_paths: set[str] = set()
    for file_record in files:
        if not isinstance(file_record, dict) or set(file_record) != {
            "relative_path",
            "node_type",
            "sha256",
        }:
            raise ValueError("Immutable Zarr metadata file fields are not exact.")
        relative_path = file_record.get("relative_path")
        if (
            type(relative_path) is not str
            or Path(relative_path).name != "zarr.json"
            or relative_path.startswith("/")
            or ".." in Path(relative_path).parts
            or relative_path in seen_paths
            or file_record.get("node_type") not in {"array", "group"}
        ):
            raise ValueError("Immutable Zarr metadata file identity is invalid.")
        _require_sha256(
            file_record.get("sha256"), label="immutable Zarr metadata digest"
        )
        seen_paths.add(relative_path)
    if files != sorted(files, key=lambda item: item["relative_path"]):
        raise ValueError("Immutable Zarr metadata files are not canonically ordered.")
    if metadata.get("root_sha256") != canonical_json_sha256(files):
        raise ValueError("Immutable Zarr metadata root is stale.")
    return metadata


def build_payload_integrity_receipt(
    run_path: str | Path,
    *,
    run_ref: str,
    decoded_copy_report: Mapping[str, Any],
    hash_workers: int = 4,
) -> dict[str, Any]:
    """Build one closed decoded-and-physical receipt for an immutable run."""

    root = Path(run_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Zarr run path is unavailable: {root}")
    canonical_ref = str(run_ref).strip()
    if not canonical_ref.startswith("/") or canonical_ref.endswith("/"):
        raise ValueError("Payload receipt run_ref must be one absolute Zarr node ref.")
    body = {
        "schema_id": PAYLOAD_INTEGRITY_RECEIPT_SCHEMA_ID,
        "schema_version": PAYLOAD_INTEGRITY_RECEIPT_SCHEMA_VERSION,
        "receipt_role": "immutable_output_payload_integrity_not_scientific_validation",
        "run_ref": canonical_ref,
        "merkle_composition": MERKLE_COMPOSITION,
        "decoded_payload": _decoded_payload_record(decoded_copy_report),
        "physical_payload": _physical_payload_record(root, workers=hash_workers),
        "immutable_metadata": _immutable_metadata_record(root),
        "metadata_scope": (
            "zarr_json_attributes_excluded_and_validated_by_stage_contract"
        ),
        "closed_array_inventory": True,
        "closed_physical_payload_inventory": True,
    }
    return {**body, "record_sha256": canonical_json_sha256(body)}


def canonical_payload_integrity_receipt(value: Any) -> dict[str, Any]:
    canonical = _canonical_copy(value, label="payload integrity receipt")
    if not isinstance(canonical, dict):
        raise ValueError("Payload integrity receipt must be a JSON object.")
    digest = canonical.pop("record_sha256", None)
    expected = {
        "schema_id",
        "schema_version",
        "receipt_role",
        "run_ref",
        "merkle_composition",
        "decoded_payload",
        "physical_payload",
        "immutable_metadata",
        "metadata_scope",
        "closed_array_inventory",
        "closed_physical_payload_inventory",
    }
    if set(canonical) != expected:
        raise ValueError("Payload integrity receipt fields are not exact.")
    if (
        canonical.get("schema_id") != PAYLOAD_INTEGRITY_RECEIPT_SCHEMA_ID
        or canonical.get("schema_version") != PAYLOAD_INTEGRITY_RECEIPT_SCHEMA_VERSION
        or canonical.get("receipt_role")
        != "immutable_output_payload_integrity_not_scientific_validation"
        or canonical.get("merkle_composition") != MERKLE_COMPOSITION
        or canonical.get("metadata_scope")
        != "zarr_json_attributes_excluded_and_validated_by_stage_contract"
        or canonical.get("closed_array_inventory") is not True
        or canonical.get("closed_physical_payload_inventory") is not True
        or not isinstance(canonical.get("run_ref"), str)
        or not canonical["run_ref"].startswith("/")
        or canonical["run_ref"].endswith("/")
        or not _is_sha256(digest)
        or digest != canonical_json_sha256(canonical)
    ):
        raise ValueError("Payload integrity receipt is unsupported or stale.")
    canonical["decoded_payload"] = _validate_decoded_payload_record(
        canonical.get("decoded_payload")
    )
    canonical["physical_payload"] = _validate_physical_payload_record(
        canonical.get("physical_payload")
    )
    canonical["immutable_metadata"] = _validate_immutable_metadata_record(
        canonical.get("immutable_metadata")
    )
    return {**canonical, "record_sha256": str(digest)}


def verify_payload_integrity_receipt(
    run_path: str | Path,
    receipt: Mapping[str, Any],
    *,
    expected_run_ref: str,
    hash_workers: int = 4,
    verify_physical_payload: bool = True,
) -> dict[str, Any]:
    """Validate the receipt and optionally rehash the installed payload files."""

    canonical = canonical_payload_integrity_receipt(receipt)
    if canonical["run_ref"] != str(expected_run_ref):
        raise ValueError("Payload integrity receipt names a different canonical run.")
    if verify_physical_payload:
        observed = _physical_payload_record(
            Path(run_path).expanduser().resolve(), workers=hash_workers
        )
        if observed != canonical["physical_payload"]:
            raise ValueError(
                "Installed physical payload differs from its integrity receipt."
            )
    observed_metadata = _immutable_metadata_record(
        Path(run_path).expanduser().resolve()
    )
    if observed_metadata != canonical["immutable_metadata"]:
        raise ValueError(
            "Installed immutable Zarr metadata differs from its integrity receipt."
        )
    return canonical


def build_payload_validation_receipt(
    integrity_receipt: Mapping[str, Any],
    *,
    scientific_manifest_schema_id: str,
    scientific_manifest_schema_version: int,
    scientific_manifest_sha256: str,
    validator_schema_id: str,
    validator_schema_version: int,
    numerical_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind one successful scientific validator to one immutable payload root."""

    integrity = canonical_payload_integrity_receipt(integrity_receipt)
    _require_sha256(
        scientific_manifest_sha256,
        label="scientific manifest digest",
    )
    if (
        type(scientific_manifest_schema_id) is not str
        or not scientific_manifest_schema_id
        or type(scientific_manifest_schema_version) is not int
        or scientific_manifest_schema_version < 1
        or type(validator_schema_id) is not str
        or not validator_schema_id
        or type(validator_schema_version) is not int
        or validator_schema_version < 1
    ):
        raise ValueError(
            "Validation receipt schema and validator identities are invalid."
        )
    policy = _canonical_copy(numerical_policy, label="numerical policy")
    if not isinstance(policy, dict) or not policy:
        raise ValueError(
            "Validation receipt numerical policy must be a nonempty object."
        )
    body = {
        "schema_id": PAYLOAD_VALIDATION_RECEIPT_SCHEMA_ID,
        "schema_version": PAYLOAD_VALIDATION_RECEIPT_SCHEMA_VERSION,
        "receipt_role": "scientific_validation_of_exact_immutable_output_payload",
        "run_ref": integrity["run_ref"],
        "integrity_receipt_sha256": integrity["record_sha256"],
        "decoded_payload_root_sha256": integrity["decoded_payload"]["root_sha256"],
        "physical_payload_root_sha256": integrity["physical_payload"]["root_sha256"],
        "immutable_metadata_root_sha256": integrity["immutable_metadata"][
            "root_sha256"
        ],
        "scientific_manifest": {
            "schema_id": scientific_manifest_schema_id,
            "schema_version": scientific_manifest_schema_version,
            "sha256": str(scientific_manifest_sha256),
        },
        "validator": {
            "schema_id": validator_schema_id,
            "schema_version": validator_schema_version,
        },
        "numerical_policy": policy,
        "numerical_policy_sha256": canonical_json_sha256(policy),
        "result": "valid",
    }
    return {**body, "record_sha256": canonical_json_sha256(body)}


def verify_payload_validation_receipt(
    value: Mapping[str, Any],
    *,
    integrity_receipt: Mapping[str, Any],
    expected_scientific_manifest_sha256: str,
    expected_validator_schema_id: str,
    expected_validator_schema_version: int,
) -> dict[str, Any]:
    canonical = _canonical_copy(value, label="payload validation receipt")
    if not isinstance(canonical, dict):
        raise ValueError("Payload validation receipt must be a JSON object.")
    digest = canonical.pop("record_sha256", None)
    expected_fields = {
        "schema_id",
        "schema_version",
        "receipt_role",
        "run_ref",
        "integrity_receipt_sha256",
        "decoded_payload_root_sha256",
        "physical_payload_root_sha256",
        "immutable_metadata_root_sha256",
        "scientific_manifest",
        "validator",
        "numerical_policy",
        "numerical_policy_sha256",
        "result",
    }
    if set(canonical) != expected_fields:
        raise ValueError("Payload validation receipt fields are not exact.")
    integrity = canonical_payload_integrity_receipt(integrity_receipt)
    if (
        canonical.get("schema_id") != PAYLOAD_VALIDATION_RECEIPT_SCHEMA_ID
        or canonical.get("schema_version") != PAYLOAD_VALIDATION_RECEIPT_SCHEMA_VERSION
        or canonical.get("receipt_role")
        != "scientific_validation_of_exact_immutable_output_payload"
        or canonical.get("result") != "valid"
        or not _is_sha256(digest)
        or digest != canonical_json_sha256(canonical)
        or canonical.get("run_ref") != integrity["run_ref"]
        or canonical.get("integrity_receipt_sha256") != integrity["record_sha256"]
        or canonical.get("decoded_payload_root_sha256")
        != integrity["decoded_payload"]["root_sha256"]
        or canonical.get("physical_payload_root_sha256")
        != integrity["physical_payload"]["root_sha256"]
        or canonical.get("immutable_metadata_root_sha256")
        != integrity["immutable_metadata"]["root_sha256"]
    ):
        raise ValueError(
            "Payload validation receipt is unsupported, stale, or unbound."
        )
    manifest = canonical.get("scientific_manifest")
    validator = canonical.get("validator")
    policy = canonical.get("numerical_policy")
    if (
        not isinstance(manifest, dict)
        or manifest.get("sha256") != expected_scientific_manifest_sha256
        or not isinstance(validator, dict)
        or validator
        != {
            "schema_id": expected_validator_schema_id,
            "schema_version": expected_validator_schema_version,
        }
        or not isinstance(policy, dict)
        or not policy
        or canonical.get("numerical_policy_sha256") != canonical_json_sha256(policy)
    ):
        raise ValueError(
            "Payload validation receipt validator or manifest binding changed."
        )
    return {**canonical, "record_sha256": str(digest)}


__all__ = [
    "DECODED_PAYLOAD_CANONICALIZATION",
    "MERKLE_COMPOSITION",
    "IMMUTABLE_METADATA_CANONICALIZATION",
    "PAYLOAD_INTEGRITY_RECEIPT_SCHEMA_ID",
    "PAYLOAD_INTEGRITY_RECEIPT_SCHEMA_VERSION",
    "PAYLOAD_VALIDATION_RECEIPT_SCHEMA_ID",
    "PAYLOAD_VALIDATION_RECEIPT_SCHEMA_VERSION",
    "decoded_payload_receipt_from_copy_report",
    "PHYSICAL_PAYLOAD_CANONICALIZATION",
    "build_payload_integrity_receipt",
    "build_payload_validation_receipt",
    "canonical_json_sha256",
    "canonical_payload_integrity_receipt",
    "verify_payload_integrity_receipt",
    "verify_payload_validation_receipt",
]
