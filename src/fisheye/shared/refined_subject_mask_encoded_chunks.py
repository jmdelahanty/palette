"""Global-grid encoded chunk helpers for refined subject-mask clip packages."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np
import zarr


GLOBAL_GRID_SCHEMA_ID = "palette_refined_subject_mask_global_chunk_grid_v1"
ENCODED_PACKAGE_SCHEMA_ID = "palette_refined_subject_mask_clip_package_v2"
ENCODED_MASK_PAYLOAD_NAME = "encoded_global_masks_roi"
ENCODED_MASK_PAYLOAD_SCHEMA_ID = "palette_refined_subject_mask_encoded_global_chunks_v1"
_GRID_FINGERPRINT_FIELDS = (
    "schema_id",
    "crop_run",
    "row_identity",
    "row_count",
    "mask_labels",
    "mask_shape",
    "dense_mask_chunks",
    "array_contract_fingerprint",
    "source_crop_row_ids_sha256",
    "destination_rows_sha256",
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, *, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _atomic_save_array(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(tmp, path)


def array_contract_from_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "zarr_format": metadata.get("zarr_format"),
        "node_type": metadata.get("node_type"),
        "shape": list(metadata.get("shape") or []),
        "data_type": metadata.get("data_type"),
        "chunk_grid": metadata.get("chunk_grid"),
        "chunk_key_encoding": metadata.get("chunk_key_encoding"),
        "fill_value": metadata.get("fill_value"),
        "codecs": metadata.get("codecs"),
        "storage_transformers": metadata.get("storage_transformers") or [],
    }


def array_contract_fingerprint(metadata: Mapping[str, Any]) -> str:
    return sha256_bytes(_canonical_json_bytes(array_contract_from_metadata(metadata)))


def _canonical_dense_mask_metadata(
    *,
    shape: Sequence[int],
    chunks: Sequence[int],
) -> dict[str, Any]:
    """Return the versioned ordinary-Zarr contract used by dense mask runs."""

    return {
        "shape": [int(value) for value in shape],
        "data_type": "uint8",
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": [int(value) for value in chunks]},
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
        "fill_value": 0,
        "codecs": [
            {"name": "bytes"},
            {"name": "zstd", "configuration": {"level": 0, "checksum": False}},
        ],
        "zarr_format": 3,
        "node_type": "array",
        "storage_transformers": [],
    }


def prepare_global_mask_chunk_grid(
    *,
    zarr_path: Path,
    crop_run: str,
    output_manifest: Path,
    mask_labels: Sequence[str],
    mask_height: int = 512,
    mask_width: int = 512,
    dense_mask_row_chunk: int = 128,
) -> dict[str, Any]:
    """Write one immutable crop-identity grid shared by all clip packagers."""

    zarr_path = zarr_path.expanduser().resolve()
    output_manifest = output_manifest.expanduser().resolve()
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    crop = root["crop_runs"][str(crop_run)]
    if "source_crop_row_ids" not in crop:
        raise ValueError(f"crop_runs/{crop_run} is missing source_crop_row_ids.")
    source_crop_row_ids = np.asarray(crop["source_crop_row_ids"][:], dtype=np.int64).reshape(-1)
    unique = np.unique(source_crop_row_ids)
    if int(unique.shape[0]) != int(source_crop_row_ids.shape[0]):
        raise ValueError(f"crop_runs/{crop_run} contains duplicate source_crop_row_ids.")
    order = np.argsort(source_crop_row_ids, kind="stable")
    sorted_ids = source_crop_row_ids[order]
    # The collection importer canonicalizes rows by sorted stable identity.  Seal
    # that exact destination order into the grid rather than inheriting whatever
    # incidental order the crop proxy currently uses.
    destination_rows = np.arange(source_crop_row_ids.shape[0], dtype=np.int64)

    ids_path = output_manifest.with_name(f"{output_manifest.stem}.source_crop_row_ids.npy")
    rows_path = output_manifest.with_name(f"{output_manifest.stem}.destination_rows.npy")
    _atomic_save_array(ids_path, sorted_ids)
    _atomic_save_array(rows_path, destination_rows)
    mask_shape = [
        int(source_crop_row_ids.shape[0]),
        int(len(mask_labels)),
        int(mask_height),
        int(mask_width),
    ]
    dense_mask_chunks = [
        int(dense_mask_row_chunk),
        1,
        int(mask_height),
        int(mask_width),
    ]
    canonical_metadata = _canonical_dense_mask_metadata(
        shape=mask_shape,
        chunks=dense_mask_chunks,
    )
    payload = {
        "schema_id": GLOBAL_GRID_SCHEMA_ID,
        "zarr_path": str(zarr_path),
        "crop_run": str(crop_run),
        "row_identity": "source_crop_row_ids",
        "row_count": int(source_crop_row_ids.shape[0]),
        "mask_labels": [str(value) for value in mask_labels],
        "mask_shape": mask_shape,
        "dense_mask_chunks": dense_mask_chunks,
        "array_contract": array_contract_from_metadata(canonical_metadata),
        "array_contract_fingerprint": array_contract_fingerprint(canonical_metadata),
        "source_crop_row_ids_path": str(ids_path),
        "source_crop_row_ids_sha256": sha256_file(ids_path),
        "destination_rows_path": str(rows_path),
        "destination_rows_sha256": sha256_file(rows_path),
    }
    payload["grid_fingerprint"] = sha256_bytes(
        _canonical_json_bytes({key: payload[key] for key in _GRID_FINGERPRINT_FIELDS})
    )
    _atomic_write_json(output_manifest, payload)
    return payload


@dataclass(frozen=True)
class GlobalMaskChunkGrid:
    manifest_path: Path
    manifest: Mapping[str, Any]
    sorted_source_crop_row_ids: np.ndarray
    destination_rows: np.ndarray

    @property
    def row_count(self) -> int:
        return int(self.manifest["row_count"])

    @property
    def mask_shape(self) -> tuple[int, int, int, int]:
        return tuple(int(value) for value in self.manifest["mask_shape"])  # type: ignore[return-value]

    @property
    def chunks(self) -> tuple[int, int, int, int]:
        return tuple(int(value) for value in self.manifest["dense_mask_chunks"])  # type: ignore[return-value]


def load_global_mask_chunk_grid(manifest_path: Path) -> GlobalMaskChunkGrid:
    manifest_path = manifest_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_id") != GLOBAL_GRID_SCHEMA_ID:
        raise ValueError(
            f"Global mask grid {manifest_path} has schema_id={manifest.get('schema_id')!r}; "
            f"expected {GLOBAL_GRID_SCHEMA_ID!r}."
        )
    expected_fingerprint = sha256_bytes(
        _canonical_json_bytes({key: manifest.get(key) for key in _GRID_FINGERPRINT_FIELDS})
    )
    if str(manifest.get("grid_fingerprint") or "") != expected_fingerprint:
        raise ValueError(f"Global mask grid manifest fingerprint mismatch: {manifest_path}")
    contract = manifest.get("array_contract")
    if not isinstance(contract, Mapping):
        raise ValueError(f"Global mask grid has no canonical array contract: {manifest_path}")
    if array_contract_fingerprint(contract) != str(manifest.get("array_contract_fingerprint") or ""):
        raise ValueError(f"Global mask grid array contract fingerprint mismatch: {manifest_path}")
    ids_path = Path(str(manifest["source_crop_row_ids_path"])).expanduser().resolve()
    rows_path = Path(str(manifest["destination_rows_path"])).expanduser().resolve()
    if sha256_file(ids_path) != str(manifest["source_crop_row_ids_sha256"]):
        raise ValueError(f"Global mask grid source-crop identity digest mismatch: {ids_path}")
    if sha256_file(rows_path) != str(manifest["destination_rows_sha256"]):
        raise ValueError(f"Global mask grid destination-row digest mismatch: {rows_path}")
    source_ids = np.load(ids_path, mmap_mode="r", allow_pickle=False)
    destination_rows = np.load(rows_path, mmap_mode="r", allow_pickle=False)
    if source_ids.shape != destination_rows.shape or int(source_ids.shape[0]) != int(manifest["row_count"]):
        raise ValueError(f"Global mask grid mapping arrays have inconsistent shapes: {manifest_path}")
    if source_ids.dtype != np.dtype(np.int64) or destination_rows.dtype != np.dtype(np.int64):
        raise ValueError(f"Global mask grid mapping arrays must use int64: {manifest_path}")
    if source_ids.size and not bool(np.all(source_ids[1:] > source_ids[:-1])):
        raise ValueError(f"Global mask grid source identities are not strictly increasing: {manifest_path}")
    expected_rows = np.arange(int(manifest["row_count"]), dtype=np.int64)
    if not np.array_equal(destination_rows, expected_rows):
        raise ValueError(f"Global mask grid destination rows are not canonical sorted ranks: {manifest_path}")
    return GlobalMaskChunkGrid(
        manifest_path=manifest_path,
        manifest=manifest,
        sorted_source_crop_row_ids=source_ids,
        destination_rows=destination_rows,
    )


def map_source_crop_rows(grid: GlobalMaskChunkGrid, source_crop_row_ids: np.ndarray) -> np.ndarray:
    values = np.asarray(source_crop_row_ids, dtype=np.int64).reshape(-1)
    indices = np.searchsorted(grid.sorted_source_crop_row_ids, values)
    valid = indices < int(grid.sorted_source_crop_row_ids.shape[0])
    if not bool(np.all(valid)):
        missing = int(values[np.flatnonzero(~valid)[0]])
        raise ValueError(f"source_crop_row_id {missing} is absent from the global mask grid.")
    matched = np.asarray(grid.sorted_source_crop_row_ids[indices], dtype=np.int64)
    equal = matched == values
    if not bool(np.all(equal)):
        missing = int(values[np.flatnonzero(~equal)[0]])
        raise ValueError(f"source_crop_row_id {missing} is absent from the global mask grid.")
    destination_rows = np.asarray(grid.destination_rows[indices], dtype=np.int64)
    if int(np.unique(destination_rows).shape[0]) != int(destination_rows.shape[0]):
        raise ValueError("Clip source_crop_row_ids map to duplicate global destination rows.")
    return destination_rows


def _contiguous_local_destination_runs(
    local_rows: np.ndarray,
    destination_rows: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    if int(local_rows.shape[0]) == 0:
        return []
    out: list[tuple[int, int, int, int]] = []
    start = 0
    while start < int(local_rows.shape[0]):
        stop = start + 1
        while (
            stop < int(local_rows.shape[0])
            and int(local_rows[stop]) == int(local_rows[stop - 1]) + 1
            and int(destination_rows[stop]) == int(destination_rows[stop - 1]) + 1
        ):
            stop += 1
        out.append(
            (
                int(local_rows[start]),
                int(local_rows[stop - 1]) + 1,
                int(destination_rows[start]),
                int(destination_rows[stop - 1]) + 1,
            )
        )
        start = stop
    return out


def _chunk_owned_ranges(destination_rows: np.ndarray, *, row_count: int, row_chunk: int) -> dict[int, list[list[int]]]:
    local_order = np.argsort(destination_rows, kind="stable").astype(np.int64, copy=False)
    ordered_dest = destination_rows[local_order]
    ranges: dict[int, list[list[int]]] = {}
    for _src_start, _src_stop, dst_start, dst_stop in _contiguous_local_destination_runs(local_order, ordered_dest):
        current = int(dst_start)
        while current < int(dst_stop):
            chunk_idx = current // int(row_chunk)
            stop = min(int(dst_stop), (chunk_idx + 1) * int(row_chunk), int(row_count))
            ranges.setdefault(chunk_idx, []).append([current, stop])
            current = stop
    return ranges


def _copy_chunk_rows(
    *,
    source: Any,
    destination: Any,
    local_order: np.ndarray,
    ordered_dest: np.ndarray,
    dst_start: int,
    dst_stop: int,
) -> None:
    tail = tuple(int(value) for value in destination.shape[1:])
    block = np.zeros((int(dst_stop) - int(dst_start), *tail), dtype=np.dtype(destination.dtype))
    left = int(np.searchsorted(ordered_dest, int(dst_start), side="left"))
    right = int(np.searchsorted(ordered_dest, int(dst_stop), side="left"))
    if right <= left:
        return
    selected_local = local_order[left:right]
    selected_dest = ordered_dest[left:right]
    for src_start, src_stop, run_dst_start, run_dst_stop in _contiguous_local_destination_runs(
        selected_local,
        selected_dest,
    ):
        rel_start = int(run_dst_start) - int(dst_start)
        rel_stop = int(run_dst_stop) - int(dst_start)
        block[rel_start:rel_stop] = source[src_start:src_stop]
    destination[dst_start:dst_stop] = block


def build_global_encoded_mask_payload(
    *,
    run_path: Path,
    grid_manifest_path: Path,
    payload_path: Path,
    copy_workers: int = 8,
) -> dict[str, Any]:
    """Re-encode one clip run locally onto the immutable global chunk grid."""

    run_path = run_path.expanduser().resolve()
    payload_path = payload_path.expanduser().resolve()
    grid = load_global_mask_chunk_grid(grid_manifest_path)
    run = zarr.open_group(str(run_path), mode="r", use_consolidated=False)
    source = run["masks_roi"]
    source_labels = [str(value) for value in run.attrs.get("mask_labels") or []]
    grid_labels = [str(value) for value in grid.manifest.get("mask_labels") or []]
    if source_labels != grid_labels:
        raise ValueError(
            f"{run_path} mask_labels {source_labels!r} do not match global grid {grid_labels!r}."
        )
    source_ids = np.asarray(run["source_crop_row_ids"][:], dtype=np.int64).reshape(-1)
    if int(source.shape[0]) != int(source_ids.shape[0]):
        raise ValueError(f"{run_path}/masks_roi row count does not match source_crop_row_ids.")
    if tuple(int(value) for value in source.shape[1:]) != tuple(grid.mask_shape[1:]):
        raise ValueError(
            f"{run_path}/masks_roi row shape {tuple(source.shape[1:])} does not match grid {grid.mask_shape[1:]}."
        )
    if tuple(int(value) for value in source.chunks) != tuple(grid.chunks):
        raise ValueError(
            f"{run_path}/masks_roi chunks {tuple(source.chunks)} do not match grid {grid.chunks}."
        )
    destination_rows = map_source_crop_rows(grid, source_ids)
    local_order = np.argsort(destination_rows, kind="stable").astype(np.int64, copy=False)
    ordered_dest = destination_rows[local_order]

    if payload_path.exists():
        shutil.rmtree(payload_path)
    payload_path.mkdir(parents=True)
    source_metadata = json.loads((run_path / "masks_roi" / "zarr.json").read_text(encoding="utf-8"))
    payload_metadata = dict(source_metadata)
    payload_metadata["shape"] = [int(value) for value in grid.mask_shape]
    payload_contract_fingerprint = array_contract_fingerprint(payload_metadata)
    grid_contract_fingerprint = str(grid.manifest["array_contract_fingerprint"])
    if payload_contract_fingerprint != grid_contract_fingerprint:
        raise ValueError(
            f"{run_path}/masks_roi encoded contract {payload_contract_fingerprint} does not match "
            f"global grid contract {grid_contract_fingerprint}."
        )
    _atomic_write_json(payload_path / "zarr.json", payload_metadata)
    destination = zarr.open_array(str(payload_path), mode="a")

    row_chunk = int(grid.chunks[0])
    chunk_indices = sorted({int(value) // row_chunk for value in destination_rows.tolist()})
    workers = max(1, int(copy_workers))

    def write_chunk(chunk_idx: int) -> None:
        dst_start = int(chunk_idx) * row_chunk
        dst_stop = min(dst_start + row_chunk, grid.row_count)
        _copy_chunk_rows(
            source=source,
            destination=destination,
            local_order=local_order,
            ordered_dest=ordered_dest,
            dst_start=dst_start,
            dst_stop=dst_stop,
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(write_chunk, chunk_idx) for chunk_idx in chunk_indices]
        for future in futures:
            future.result()

    owned_ranges = _chunk_owned_ranges(destination_rows, row_count=grid.row_count, row_chunk=row_chunk)
    chunk_records: list[dict[str, Any]] = []
    object_count = 0
    stored_bytes = 0
    for chunk_idx in chunk_indices:
        dst_start = int(chunk_idx) * row_chunk
        dst_stop = min(dst_start + row_chunk, grid.row_count)
        ranges = owned_ranges[int(chunk_idx)]
        coverage = sum(int(stop) - int(start) for start, stop in ranges)
        objects: list[dict[str, Any]] = []
        chunk_root = payload_path / "c" / str(chunk_idx)
        if chunk_root.is_dir():
            for object_path in sorted(path for path in chunk_root.rglob("*") if path.is_file()):
                size = int(object_path.stat().st_size)
                objects.append(
                    {
                        "path": object_path.relative_to(payload_path).as_posix(),
                        "size_bytes": size,
                        "sha256": sha256_file(object_path),
                    }
                )
                object_count += 1
                stored_bytes += size
        chunk_records.append(
            {
                "row_chunk_index": int(chunk_idx),
                "dst_start": dst_start,
                "dst_stop": dst_stop,
                "owned_ranges": ranges,
                "owned_row_count": int(coverage),
                "complete": int(coverage) == int(dst_stop - dst_start),
                "objects": objects,
            }
        )

    return {
        "schema_id": ENCODED_MASK_PAYLOAD_SCHEMA_ID,
        "payload_path": ENCODED_MASK_PAYLOAD_NAME,
        "grid_manifest": str(grid.manifest_path),
        "grid_fingerprint": str(grid.manifest["grid_fingerprint"]),
        "array_contract": array_contract_from_metadata(payload_metadata),
        "array_contract_fingerprint": payload_contract_fingerprint,
        "source_row_count": int(source_ids.shape[0]),
        "global_row_min": int(destination_rows.min()) if destination_rows.size else None,
        "global_row_max": int(destination_rows.max()) if destination_rows.size else None,
        "global_row_count": int(grid.row_count),
        "row_chunk_count": int(len(chunk_records)),
        "complete_row_chunk_count": int(sum(bool(row["complete"]) for row in chunk_records)),
        "fragment_row_chunk_count": int(sum(not bool(row["complete"]) for row in chunk_records)),
        "object_count": int(object_count),
        "stored_bytes": int(stored_bytes),
        "requested_copy_workers": int(copy_workers),
        "effective_copy_workers": workers,
        "write_ownership": "one_thread_owns_all_physical_objects_for_each_row_chunk",
        "chunks": chunk_records,
    }
