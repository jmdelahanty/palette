"""Sealed reference to an admitted unified H5, and its only reader.

The raw H5 stays in the recording's ``raw/`` tree as the single primary source
(its finalization receipt binds its exact bytes). The analysis Zarr stores a
reference: the H5 path relative to the store, its size and mtime, the admission
claims, a snapshot of every attribute, and SHA-256 digests of each block of
every fixed-width dataset on admission's own block grid. Readers check size and
mtime on open and verify exactly the blocks a request touches.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import os
from pathlib import Path
from uuid import uuid4

import h5py
import numpy as np
import zarr

from fisheye.shared.zarr.array_contracts import UINT8, ArrayContract
from fisheye.shared.zarr.array_factory import create_array_from_plan
from fisheye.shared.zarr.storage_intent import AccessPattern, WriteMode
from fisheye.shared.zarr.storage_planner import plan_storage
from fisheye.shared.zarr.storage_profiles import get_storage_profile
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_NAME_ATTR,
    RUN_STATUS_COMPLETE,
)

from .common import (
    MAX_JSON_BYTES,
    PROFILE,
    AdmissionScan,
    canonical_json,
    contract_errors,
    digest,
    parse_json,
    require,
)
from .hdf5_types import block_rows, dataset_bytes, iter_blocks, iter_payload, type_descriptor
from .metadata import native_attributes

REFERENCE_SCHEMA = "palette.unified_h5_source_reference"
REFERENCE_VERSION = 1
REFERENCE_ARRAY = "source_reference_json_utf8"
DIGEST_ARRAY = "block_digests"
REFERENCE_DIGEST_ATTR = "source_reference_sha256"
_BYTES = ArrayContract(
    schema_id="palette.unified_h5_source_reference_bytes",
    schema_version=1,
    dtype=UINT8,
    shape_template=("n_bytes",),
    axis_names=("byte",),
    description="Canonical reference JSON, or concatenated 32-byte block SHA-256 digests.",
)


def new_native_run_name() -> str:
    """Fresh immutable run name for a unified source reference."""

    return (
        "unified_native_"
        + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_")
        + uuid4().hex[:12]
    )


def _write_bytes(group, name: str, data: bytes) -> None:
    plan = plan_storage(
        _BYTES.storage_intent(
            shape=(len(data),),
            access=AccessPattern.WINDOWED,
            write_mode=WriteMode.IMMUTABLE,
            whole_shard_writes=False,
        ),
        get_storage_profile("scratch_compute_v1"),
    )
    array = create_array_from_plan(group, name=name, contract=_BYTES, plan=plan, fill_value=0)
    array[:] = np.frombuffer(data, dtype=np.uint8)


def _block_bytes(dataset, block) -> bytes:
    return np.asarray(block, dtype=dataset.dtype).tobytes(order="C")


def _is_blocked(dataset) -> bool:
    return dataset.shape != () and not dataset.dtype.hasobject


def _whole_digest(dataset) -> str:
    hasher = sha256()
    for part in iter_payload(dataset):
        hasher.update(part)
    return hasher.hexdigest()


def seal_reference(run, h5, *, admission, scan: AdmissionScan, source_h5, zarr_path) -> str:
    """Write the reference arrays into ``run`` and return the record digest.

    The caller stores the digest as ``REFERENCE_DIGEST_ATTR`` through its
    ownership-checked attribute writer.

    Block digests come from the admission scan; a fixed-width dataset the scan
    did not fully cover is hashed here (small datasets only, in practice).
    """

    source = Path(source_h5).expanduser().resolve(strict=True)
    store = Path(zarr_path).resolve()
    require(source.is_relative_to(store.parent.parent), "unified_source_outside_recording")
    datasets, digests = {}, []
    for path, kind in sorted(admission.node_kinds.items()):
        if kind != "dataset":
            continue
        dataset = h5[path]
        entry = {"type": type_descriptor(dataset.id.get_type()), "shape": list(dataset.shape)}
        if _is_blocked(dataset):
            rows = block_rows(dataset)
            recorded = scan.blocks.get(dataset.name, {})
            starts = range(0, dataset.shape[0], rows)
            if set(recorded) != set(starts):
                recorded = {
                    start: sha256(_block_bytes(dataset, block)).hexdigest()
                    for start, block in iter_blocks(dataset)
                }
            entry.update(mode="blocks", rows_per_block=rows, first_block=len(digests))
            digests.extend(recorded[start] for start in starts)
        else:
            entry.update(mode="whole", sha256=_whole_digest(dataset))
        datasets[path] = entry
    digest_bytes = b"".join(bytes.fromhex(value) for value in digests)
    stat = source.stat()
    record = {
        "schema_id": REFERENCE_SCHEMA,
        "schema_version": REFERENCE_VERSION,
        "source": {
            "relative_path": os.path.relpath(source, store),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
        "admission": admission.manifest_claims(),
        "datasets": datasets,
        "block_digests_sha256": sha256(digest_bytes).hexdigest(),
        "attributes": {
            path: native_attributes(h5[path])
            for path in ["/", *sorted(admission.node_kinds)]
        },
    }
    raw = canonical_json(record)
    require(len(raw) <= MAX_JSON_BYTES, "unified_reference_byte_budget")
    _write_bytes(run, REFERENCE_ARRAY, raw)
    _write_bytes(run, DIGEST_ARRAY, digest_bytes)
    return digest(raw)


class UnifiedSource:
    """Verified, read-only access to one admitted unified H5."""

    def __init__(self, root, run_name: str):
        require(
            isinstance(run_name, str) and run_name not in ("", ".", "..") and "/" not in run_name,
            "unified_run_name_invalid",
        )
        run = root[f"analysis/stimulus_runs/{run_name}"]
        # A stale consolidated-metadata view must not hide a direct change.
        store = Path(root.store.root).resolve()
        direct = zarr.open_group(str(store), mode="r", use_consolidated=False)[
            f"analysis/stimulus_runs/{run_name}"
        ]
        require(dict(direct.attrs) == dict(run.attrs), "unified_reference_metadata_split")
        require(
            run.attrs.get(RUN_COMPLETION_STATUS_ATTR) == RUN_STATUS_COMPLETE
            and run.attrs.get(RUN_NAME_ATTR) == run_name
            and run.attrs.get("stage_selector_eligible") is False
            and run.attrs.get("source_profile") == PROFILE
            and run.attrs.get("unified_reference_schema") == REFERENCE_SCHEMA
            and run.attrs.get("unified_reference_version") == REFERENCE_VERSION,
            "unified_reference_not_complete_or_ineligible",
        )
        require(run[REFERENCE_ARRAY].shape[0] <= MAX_JSON_BYTES, "unified_reference_byte_budget")
        raw = bytes(np.asarray(run[REFERENCE_ARRAY][:], dtype=np.uint8))
        require(digest(raw) == run.attrs.get(REFERENCE_DIGEST_ATTR), "unified_reference_digest_mismatch")
        self._record = parse_json(raw, label="unified_source_reference", canonical=True)
        require(
            self._record.get("schema_id") == REFERENCE_SCHEMA
            and self._record.get("schema_version") == REFERENCE_VERSION,
            "unified_reference_schema",
        )
        digests = bytes(np.asarray(run[DIGEST_ARRAY][:], dtype=np.uint8))
        require(
            sha256(digests).hexdigest() == self._record["block_digests_sha256"],
            "unified_block_digests_mismatch",
        )
        self._digests = digests
        self.reference_sha256 = run.attrs[REFERENCE_DIGEST_ATTR]
        source = self._record["source"]
        path = (store / source["relative_path"]).resolve()
        require(path.is_relative_to(store.parent.parent), "unified_source_outside_recording")
        require(path.is_file(), f"unified_source_missing:{path}")
        stat = path.stat()
        require(
            (stat.st_size, stat.st_mtime_ns) == (source["size_bytes"], source["mtime_ns"]),
            f"unified_source_changed:{path}; re-admit it with a new run name",
        )
        self.source_path = path

    @property
    def admission(self) -> dict:
        return dict(self._record["admission"])

    def typed_attributes(self, path: str = "/") -> dict:
        require(path in self._record["attributes"], "unified_attribute_node_missing")
        return dict(self._record["attributes"][path])

    def _entry(self, h5, path: str):
        entry = self._record["datasets"].get(path)
        require(entry is not None, f"unified_dataset_not_admitted:{path}")
        dataset = h5[path]
        require(
            type_descriptor(dataset.id.get_type()) == entry["type"]
            and list(dataset.shape) == entry["shape"],
            f"unified_dataset_layout_changed:{path}",
        )
        return dataset, entry

    def _verified_block(self, dataset, entry, index: int):
        rows = entry["rows_per_block"]
        block = dataset[index * rows : min((index + 1) * rows, dataset.shape[0])]
        offset = (entry["first_block"] + index) * 32
        require(
            sha256(_block_bytes(dataset, block)).digest() == self._digests[offset : offset + 32],
            f"unified_block_digest_mismatch:{dataset.name}:{index}",
        )
        return block

    def read_table(self, path: str, start: int = 0, stop: int | None = None) -> np.ndarray:
        with h5py.File(self.source_path, "r") as h5:
            dataset, entry = self._entry(h5, path)
            require(entry["mode"] == "blocks", f"unified_dataset_not_tabular:{path}")
            total = dataset.shape[0]
            stop = total if stop is None else stop
            require(0 <= start <= stop <= total, "unified_row_range_invalid")
            if start == stop:
                return dataset[0:0]
            rows = entry["rows_per_block"]
            first, last = start // rows, (stop - 1) // rows
            blocks = [self._verified_block(dataset, entry, i) for i in range(first, last + 1)]
            joined = np.concatenate(blocks) if len(blocks) > 1 else blocks[0]
            return joined[start - first * rows : stop - first * rows]

    def read_json(self, path: str) -> dict:
        with h5py.File(self.source_path, "r") as h5:
            dataset, entry = self._entry(h5, path)
            require(
                entry["mode"] == "whole" and _whole_digest(dataset) == entry["sha256"],
                f"unified_dataset_digest_mismatch:{path}",
            )
            return parse_json(dataset_bytes(dataset), label=path)

    def verify(self) -> None:
        """Stream and check every block and small dataset (integrity sweeps)."""

        with h5py.File(self.source_path, "r") as h5:
            for path in self._record["datasets"]:
                dataset, entry = self._entry(h5, path)
                if entry["mode"] == "blocks":
                    for index in range(-(-dataset.shape[0] // entry["rows_per_block"])):
                        self._verified_block(dataset, entry, index)
                else:
                    require(_whole_digest(dataset) == entry["sha256"], f"unified_dataset_digest_mismatch:{path}")


@contract_errors
def open_unified_source(root, *, run_name: str) -> UnifiedSource:
    return UnifiedSource(root, run_name)
