from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.shared.mask_bitpack import pack_binary_mask_stack, packed_width_bytes, unpack_binary_mask_stack
from fisheye.shared.mask_store import (
    open_mask_store,
    validate_bitpacked_mask_store_against_dense,
    write_bitpacked_mask_store_from_dense,
)


def test_pack_unpack_binary_mask_stack_preserves_non_byte_aligned_width() -> None:
    masks = np.zeros((3, 2, 5, 10), dtype=np.uint8)
    masks[0, 0, 1:4, 2:7] = 1
    masks[1, 1, :, 9] = 1
    masks[2, :, 0, 0] = 2

    packed = pack_binary_mask_stack(masks)
    unpacked = unpack_binary_mask_stack(packed, logical_width=masks.shape[-1])

    assert packed.shape == (3, 2, 5, packed_width_bytes(10))
    np.testing.assert_array_equal(unpacked, (masks > 0).astype(np.uint8))


def test_bitpacked_mask_store_roundtrip_and_dense_reader(tmp_path: Path) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    run = root.create_group("refined_subject_masks_runs").create_group("run_001")
    labels = ("subject_body", "eye_left", "eye_right", "swim_bladder")
    run.attrs["mask_labels"] = list(labels)
    masks = np.zeros((4, len(labels), 6, 10), dtype=np.uint8)
    masks[0, 0, 1:4, 2:5] = 1
    masks[1, 1, 2:5, 5:9] = 1
    masks[2, 2, 0:2, 8:10] = 1
    masks[3, 3, 4:6, 1:4] = 1
    dense = run.create_array("masks_roi", data=masks, chunks=(2, 1, 6, 10), overwrite=True)

    summary = write_bitpacked_mask_store_from_dense(
        run,
        dense,
        component_names=labels,
        encode_row_chunk_size=2,
        validation_mode="full",
    )

    assert summary["encoding"] == "bitpacked_binary_v1"
    assert summary["mask_bitpacked_validation"]["status"] == "passed"
    assert summary["requested_encode_row_chunk_size"] == 2
    assert summary["encode_row_chunk_size"] == 4
    assert summary["roundtrip_validation"] == {
        "status": "passed",
        "rows_checked": 4,
        "channels_checked": 4,
        "chunks_checked": 1,
        "row_chunk_size": 4,
    }
    assert (
        summary["roundtrip_validation_strategy"]
        == "write_readback_against_resident_dense_v1"
    )
    assert run.attrs["mask_store_encodings"] == ["dense_uint8", "bitpacked_binary_v1"]

    store = open_mask_store(run, prefer="bitpacked")
    np.testing.assert_array_equal(store.read_dense(), masks)
    np.testing.assert_array_equal(store.read_dense(rows=[3, 1], channels=["swim_bladder", "eye_left"]), masks[[3, 1]][:, [3, 1]])

    validation = validate_bitpacked_mask_store_against_dense(run, dense, row_chunk_size=2)
    assert validation["status"] == "passed"


def test_full_bitpacked_write_reads_each_physical_row_unit_once(
    tmp_path: Path,
) -> None:
    class _CountingDense:
        def __init__(self, values: np.ndarray) -> None:
            self.values = values
            self.shape = values.shape
            self.reads: list[tuple[int, int]] = []

        def __getitem__(self, rows: slice) -> np.ndarray:
            assert isinstance(rows, slice)
            self.reads.append((int(rows.start or 0), int(rows.stop or 0)))
            return self.values[rows]

    root = zarr.open_group(str(tmp_path / "analysis.zarr"), mode="w")
    run = root.create_group("refined_subject_masks_runs").create_group("run_001")
    labels = ("subject_body", "eye_left", "eye_right", "swim_bladder")
    run.attrs["mask_labels"] = list(labels)
    dense = _CountingDense(np.ones((1025, 4, 2, 9), dtype=np.uint8))

    summary = write_bitpacked_mask_store_from_dense(
        run,
        dense,
        component_names=labels,
        encode_row_chunk_size=256,
        storage_row_chunk_size=512,
        validation_mode="full",
    )

    assert dense.reads == [(0, 512), (512, 1024), (1024, 1025)]
    assert summary["requested_encode_row_chunk_size"] == 256
    assert summary["encode_row_chunk_size"] == 512
    assert summary["roundtrip_validation"]["chunks_checked"] == 3
