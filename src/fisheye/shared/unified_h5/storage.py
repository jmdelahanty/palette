"""Lossless native candidate storage and verified, selector-ineligible reads.

The payload encoding is new Palette storage grammar, not an H5 file-byte copy or
a normalized v5/v6 coordinate product. Logical H5 paths/types and all attribute
bits are preserved. No source file is needed to read a completed candidate.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import math

import numpy as np
import zarr

from fisheye.shared.run_provenance import RUN_PROVENANCE_ATTR, validate_run_provenance
from fisheye.shared.zarr.array_factory import (
    validate_array_metadata_declaration_from_plan,
)
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_NAME_ATTR,
    RUN_STATUS_COMPLETE,
)

from .common import (
    BLOCK_BYTES,
    MAX_DATASET_BYTES,
    MAX_JSON_BYTES,
    MAX_OBJECTS,
    PROFILE,
    canonical_json,
    contract_errors,
    digest,
    exact_keys,
    internal_path,
    parse_json,
    require,
    uint64,
)
from .hdf5_types import check_dataset_budget, dtype_from_descriptor, iter_payload
from .metadata import native_attributes
from .storage_schema import (
    MANIFEST_CONTRACT,
    PAYLOAD_CONTRACT,
    STORAGE_SCHEMA,
    STORAGE_VERSION,
    create_bytes,
    payload_plan,
)

MANIFEST_ARRAY = "native_manifest_json_utf8"
MANIFEST_DIGEST_ATTR = "native_manifest_sha256"
NODE_DIGEST_ATTR = "native_h5_node_sha256"
OWNER_ATTR = "stimulus_publication_owner_uuid"
NATIVE_ROOT = "native_h5"


def inspect_native_inventory(h5, node_kinds):
    """Preflight metadata and payload encoding budgets before destination writes."""
    result, total = {}, 0
    for path, kind in sorted(node_kinds.items()):
        node = h5[path]
        descriptor = {"kind": kind, "attributes": native_attributes(node)}
        if kind == "dataset":
            spec = check_dataset_budget(node)
            hasher, size = sha256(), 0
            for block in iter_payload(node):
                size += len(block)
                require(size <= MAX_DATASET_BYTES, "native_payload_byte_budget")
                hasher.update(block)
            descriptor.update(
                type=spec,
                shape=list(node.shape),
                size_bytes=size,
                payload_sha256="sha256:" + hasher.hexdigest(),
                encoding=(
                    "u64_le_length_prefixed_strings"
                    if spec["class"] == "string" and spec["variable_length"]
                    else "packed_le_bytes"
                ),
            )
        total += len(canonical_json(descriptor)) + len(path.encode("utf-8")) + 8
        require(total <= MAX_JSON_BYTES // 2, "native_inventory_metadata_budget")
        result[path] = descriptor
    return result


def _chunks_of_bytes(parts):
    """Reblock native rows into <=1 MiB writes without retaining the dataset."""
    for part in parts:
        for start in range(0, len(part), BLOCK_BYTES):
            yield memoryview(part)[start : start + BLOCK_BYTES]


def _write_payload(array, parts, assert_owner):
    hasher, offset = sha256(), 0
    for data in _chunks_of_bytes(parts):
        assert_owner()
        values = np.frombuffer(data, dtype="u1")
        array[offset : offset + len(values)] = values
        hasher.update(data)
        offset += len(values)
    assert_owner()
    require(offset == array.shape[0], "native_copy_length_mismatch")
    return "sha256:" + hasher.hexdigest()


def write_native_candidate(
    h5, run, *, admission, inventory, run_name, owner, provenance, assert_owner
):
    manifest = {
        "schema_id": STORAGE_SCHEMA,
        "schema_version": STORAGE_VERSION,
        "run_name": run_name,
        "publication_owner_uuid": owner,
        "admission": admission.manifest_claims(),
        "finalization_receipt": admission.finalization_receipt,
        "run_provenance_sha256": digest(canonical_json(provenance)),
        "nodes": inventory,
    }
    data = canonical_json(manifest)
    require(len(data) <= MAX_JSON_BYTES, "native_manifest_byte_budget")
    assert_owner()
    native = run.create_group(NATIVE_ROOT)
    for path, descriptor in inventory.items():
        assert_owner()
        group = native if path == "/" else native.require_group(path[1:])
        assert_owner()
        group.attrs[NODE_DIGEST_ATTR] = digest(canonical_json(descriptor))
        if descriptor["kind"] == "dataset":
            assert_owner()
            array = create_bytes(group, "payload", descriptor["size_bytes"])
            observed = _write_payload(array, iter_payload(h5[path]), assert_owner)
            require(
                observed == descriptor["payload_sha256"],
                f"native_copy_source_changed:{path}",
            )
    assert_owner()
    array = create_bytes(run, MANIFEST_ARRAY, len(data), contract=MANIFEST_CONTRACT)
    _write_payload(array, (data,), assert_owner)
    return digest(data)


def _validate_byte_array(array, size, *, contract=PAYLOAD_CONTRACT):
    require(isinstance(array, zarr.Array), "native_payload_array_missing")
    errors = validate_array_metadata_declaration_from_plan(
        parse_json(
            canonical_json(array.metadata.to_dict()), label="zarr_array_metadata"
        ),
        contract=contract,
        plan=payload_plan(size, contract),
        fill_value=0,
    )
    require(not errors, "native_payload_metadata_mismatch:" + ";".join(errors))


def _read_bounded_array(array, maximum):
    require(
        array.ndim == 1 and array.dtype == np.dtype("u1") and array.shape[0] <= maximum,
        "native_array_budget_or_type",
    )
    return b"".join(
        array[start : start + BLOCK_BYTES].tobytes()
        for start in range(0, array.shape[0], BLOCK_BYTES)
    )


def _validate_node_inventory(native, inventory, *, direct=None):
    require(
        type(inventory) is dict and len(inventory) <= MAX_OBJECTS and "/" in inventory,
        "native_inventory_invalid",
    )
    observed = {}

    def visit(group, path):
        require(
            len(observed) <= MAX_OBJECTS and len(path.split("/")) <= 35,
            "native_tree_budget",
        )
        observed[path] = "group"
        descriptor = inventory.get(path)
        require(type(descriptor) is dict, f"native_unlisted_node:{path}")
        attrs = {NODE_DIGEST_ATTR: digest(canonical_json(descriptor))}
        require(dict(group.attrs) == attrs, f"native_node_metadata_mismatch:{path}")
        if direct is not None:
            fresh = direct if path == "/" else direct[path[1:]]
            declared = {
                k: v
                for k, v in group.metadata.to_dict().items()
                if k != "consolidated_metadata"
            }
            current = {
                k: v
                for k, v in fresh.metadata.to_dict().items()
                if k != "consolidated_metadata"
            }
            require(
                declared == current and set(group.keys()) == set(fresh.keys()),
                f"native_consolidated_metadata_stale:{path}",
            )
        expected_fields = {"kind", "attributes"}
        if descriptor["kind"] == "dataset":
            expected_fields |= {
                "type",
                "shape",
                "size_bytes",
                "payload_sha256",
                "encoding",
            }
            require(set(group.keys()) == {"payload"}, f"native_dataset_children:{path}")
            size = uint64(descriptor["size_bytes"], "native_payload_size")
            require(size <= MAX_DATASET_BYTES, "native_payload_byte_budget")
            array = group["payload"]
            _validate_byte_array(array, size)
            if direct is not None:
                require(
                    array.metadata.to_dict() == fresh["payload"].metadata.to_dict(),
                    f"native_consolidated_payload_stale:{path}",
                )
            hasher = sha256()
            for start in range(0, size, BLOCK_BYTES):
                hasher.update(array[start : start + BLOCK_BYTES].tobytes())
            require(
                "sha256:" + hasher.hexdigest() == descriptor["payload_sha256"],
                f"native_payload_digest_mismatch:{path}",
            )
            dtype = dtype_from_descriptor(descriptor["type"])
            shape = descriptor["shape"]
            require(
                type(shape) is list
                and len(shape) <= 8
                and all(type(value) is int and value >= 0 for value in shape),
                "native_shape_invalid",
            )
            if not dtype.hasobject:
                require(
                    descriptor["encoding"] == "packed_le_bytes"
                    and math.prod(shape) * dtype.itemsize == size,
                    "native_packed_layout_mismatch",
                )
            else:
                require(
                    descriptor["encoding"] == "u64_le_length_prefixed_strings",
                    "native_string_encoding",
                )
        else:
            require(
                descriptor["kind"] == "group" and len(list(group.array_keys())) == 0,
                "native_group_kind",
            )
            for name in group.group_keys():
                child = path.rstrip("/") + "/" + name
                internal_path(child)
                visit(group[name], child)
        exact_keys(descriptor, expected_fields, "native_node_descriptor")

    visit(native, "/")
    require(set(observed) == set(inventory), "native_inventory_not_closed")


@dataclass(frozen=True)
class UnifiedStimulusCandidate:
    """Verified immutable native facts; never a production selector supplier.

    Reads revalidate touched payloads. The caller must not mutate the archive;
    fresh opens are required after any external generation change.
    """

    _run: object
    _manifest: dict

    @property
    def admission(self):
        return parse_json(
            canonical_json(self._manifest["admission"]), label="native_admission"
        )

    def read_table(self, path, start=0, stop=None):
        path = internal_path(path)
        descriptor = self._manifest["nodes"].get(path)
        require(
            descriptor is not None
            and descriptor["kind"] == "dataset"
            and descriptor["type"]["class"] == "packed_compound"
            and len(descriptor["shape"]) == 1,
            f"native_table_required:{path}",
        )
        count = descriptor["shape"][0]
        stop = count if stop is None else stop
        require(
            type(start) is int and type(stop) is int and 0 <= start <= stop <= count,
            "native_table_slice_invalid",
        )
        # Bounded producer datasets allow a full payload hash before facts are
        # returned. Do not reuse a stale receipt merely for a faster row read.
        raw = self.read_payload(path)
        dtype = dtype_from_descriptor(descriptor["type"])
        return np.frombuffer(
            raw, dtype=dtype, count=stop - start, offset=start * dtype.itemsize
        ).copy()

    def read_payload(self, path):
        self._assert_generation()
        path = internal_path(path)
        descriptor = self._manifest["nodes"].get(path)
        require(
            descriptor is not None and descriptor["kind"] == "dataset",
            f"native_dataset_required:{path}",
        )
        array = self._run[NATIVE_ROOT + path + "/payload"]
        _validate_byte_array(array, descriptor["size_bytes"])
        raw = _read_bounded_array(array, MAX_DATASET_BYTES)
        require(
            digest(raw) == descriptor["payload_sha256"],
            f"native_payload_digest_mismatch:{path}",
        )
        self._assert_generation()
        return raw

    def _assert_generation(self):
        fresh = zarr.open_group(
            store=self._run.store, path=self._run.path, mode="r", use_consolidated=False
        )
        require(
            fresh.attrs.get(RUN_COMPLETION_STATUS_ATTR) == RUN_STATUS_COMPLETE
            and fresh.attrs.get("stage_selector_eligible") is False
            and fresh.attrs.get(OWNER_ATTR) == self._manifest["publication_owner_uuid"]
            and fresh.attrs.get(MANIFEST_DIGEST_ATTR)
            == digest(canonical_json(self._manifest)),
            "native_candidate_generation_changed",
        )

    def read_dataset(self, path):
        """Return exact native dtype/shape, including scalar or VL byte strings."""
        raw = self.read_payload(path)
        descriptor = self._manifest["nodes"][path]
        dtype, shape = dtype_from_descriptor(descriptor["type"]), tuple(
            descriptor["shape"]
        )
        if not dtype.hasobject:
            return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
        result, offset = np.empty(shape, dtype=object), 0
        for index in np.ndindex(shape):
            require(offset + 8 <= len(raw), "native_string_length_missing")
            length = int.from_bytes(raw[offset : offset + 8], "little")
            offset += 8
            require(
                length <= MAX_JSON_BYTES and offset + length <= len(raw),
                "native_string_length_invalid",
            )
            result[index] = raw[offset : offset + length]
            offset += length
        require(offset == len(raw), "native_string_trailing_bytes")
        return result

    def read_json(self, path):
        values = self.read_dataset(path)
        if values.ndim == 1 and values.dtype == np.dtype("u1"):
            raw = values.tobytes()
        else:
            require(
                values.shape == () and values.dtype.kind in ("S", "O"),
                "native_json_encoding",
            )
            raw = bytes(values[()])
        return parse_json(raw, label=path)

    def typed_attributes(self, path="/"):
        """Return copied HDF5 attribute descriptors with exact raw value bytes."""
        self._assert_generation()
        require(path in self._manifest["nodes"], "native_attribute_node_missing")
        return deepcopy(self._manifest["nodes"][path]["attributes"])


def _load_candidate(root, run_name, *, published):
    require(
        isinstance(run_name, str)
        and run_name not in ("", ".", "..")
        and "/" not in run_name,
        "native_run_name_invalid",
    )
    path = f"analysis/stimulus_runs/{run_name}"
    run = root[path]
    require(
        run.attrs.get(RUN_COMPLETION_CONTRACT_ATTR) == RUN_COMPLETION_CONTRACT
        and run.attrs.get(RUN_COMPLETION_STATUS_ATTR) == RUN_STATUS_COMPLETE
        and run.attrs.get(RUN_NAME_ATTR) == run_name
        and run.attrs.get("stage_selector_eligible") is False
        and run.attrs.get("source_profile") == PROFILE
        and run.attrs.get("native_storage_schema") == STORAGE_SCHEMA
        and run.attrs.get("native_storage_version") == STORAGE_VERSION,
        "native_candidate_not_complete_or_ineligible",
    )
    array = run[MANIFEST_ARRAY]
    raw = _read_bounded_array(array, MAX_JSON_BYTES)
    _validate_byte_array(array, len(raw), contract=MANIFEST_CONTRACT)
    require(
        digest(raw) == run.attrs.get(MANIFEST_DIGEST_ATTR),
        "native_manifest_digest_mismatch",
    )
    manifest = parse_json(raw, label="native_manifest", canonical=True)
    exact_keys(
        manifest,
        (
            "schema_id",
            "schema_version",
            "run_name",
            "publication_owner_uuid",
            "admission",
            "finalization_receipt",
            "run_provenance_sha256",
            "nodes",
        ),
        "native_manifest",
    )
    require(
        manifest["schema_id"] == STORAGE_SCHEMA
        and type(manifest["schema_version"]) is int
        and manifest["schema_version"] == STORAGE_VERSION
        and manifest["run_name"] == run_name
        and manifest["publication_owner_uuid"] == run.attrs.get(OWNER_ATTR),
        "native_manifest_identity_mismatch",
    )
    admission = manifest["admission"]
    require(
        admission["profile"] == PROFILE
        and admission["selector_eligible"] is False
        and admission["source_sha256"]
        == manifest["finalization_receipt"]["contract"]["h5_artifact"]["sha256"]
        and admission["finalization_receipt_sha256"]
        == digest(canonical_json(manifest["finalization_receipt"])),
        "native_admission_binding",
    )
    provenance = run.attrs.get(RUN_PROVENANCE_ATTR)
    require(
        validate_run_provenance(provenance).valid
        and digest(canonical_json(provenance)) == manifest["run_provenance_sha256"],
        "native_provenance_binding",
    )
    direct_native = None
    if published:
        require(
            root.metadata.consolidated_metadata is not None,
            "native_published_consolidation_missing",
        )
        direct = zarr.open_group(
            store=root.store, path=root.path, mode="r", use_consolidated=False
        )
        fresh = direct[path]
        require(
            dict(fresh.attrs) == dict(run.attrs)
            and fresh[MANIFEST_ARRAY].metadata.to_dict() == array.metadata.to_dict(),
            "native_published_metadata_stale",
        )
        direct_native = fresh[NATIVE_ROOT]
    _validate_node_inventory(run[NATIVE_ROOT], manifest["nodes"], direct=direct_native)
    return UnifiedStimulusCandidate(run, manifest)


def verify_unpublished_native_candidate(root, *, run_name):
    """Writer-only completion verification before final root consolidation."""
    return _load_candidate(root, run_name, published=False)


@contract_errors
def load_unified_stimulus_candidate(root, *, run_name):
    """Open only a verified immutable native candidate, by explicit run name."""
    return _load_candidate(root, run_name, published=True)
