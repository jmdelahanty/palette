from __future__ import annotations

import numpy as np
import zarr
from zarr.core.dtype import VariableLengthUTF8

from fisheye.shared.detect_reason_codec import (
    decode_reason_bytes,
    open_mutable_reason_column,
    read_reason_labels,
    update_reason_rows,
    write_reason_columns,
)


def _legacy_reason(group: zarr.Group, labels: list[str]) -> None:
    array = group.create_array(
        "reason",
        shape=(len(labels),),
        chunks=(max(1, len(labels)),),
        dtype=VariableLengthUTF8(),
        fill_value="",
        overwrite=True,
    )
    array[:] = np.asarray(labels, dtype=object)


def test_write_reason_columns_retires_legacy_mirror(tmp_path) -> None:
    group = zarr.open_group(store=tmp_path / "reason.zarr", mode="w")
    _legacy_reason(group, ["stale", "stale"])

    fields = write_reason_columns(
        group,
        np.asarray(["clean", "manual_correction"], dtype=object),
        chunk_size=2,
        overwrite=True,
    )

    assert fields == ["reason_bytes"]
    assert "reason" not in group
    assert decode_reason_bytes(group["reason_bytes"][:]).tolist() == [
        "clean",
        "manual_correction",
    ]
    assert group.attrs["reason_authority"] == "reason_bytes"
    assert group.attrs["reason_fallback_order"] == ["reason_bytes", "detection_source"]


def test_read_reason_labels_preserves_legacy_read_compatibility(tmp_path) -> None:
    group = zarr.open_group(store=tmp_path / "legacy.zarr", mode="w")
    _legacy_reason(group, ["clean", "legacy_review"])

    assert read_reason_labels(group).tolist() == ["clean", "legacy_review"]


def test_update_reason_rows_canonicalizes_dual_column_group(tmp_path) -> None:
    group = zarr.open_group(store=tmp_path / "update.zarr", mode="w")
    group.create_array("frame_indices", data=np.arange(3, dtype=np.int32), chunks=(3,))
    write_reason_columns(
        group,
        np.asarray(["clean", "clean", "clean"], dtype=object),
        chunk_size=3,
        overwrite=True,
    )
    _legacy_reason(group, ["stale", "stale", "stale"])

    update_reason_rows(
        group,
        np.asarray([1], dtype=np.int64),
        np.asarray(["a_reason_longer_than_the_existing_width_" * 2], dtype=object),
    )

    assert "reason" not in group
    assert read_reason_labels(group).tolist() == [
        "clean",
        "a_reason_longer_than_the_existing_width_a_reason_longer_than_the_existing_width_",
        "clean",
    ]


def test_mutable_reason_column_writes_slices_to_reason_bytes(tmp_path) -> None:
    group = zarr.open_group(store=tmp_path / "mutable.zarr", mode="w")
    _legacy_reason(group, ["clean", "clean", "clean"])

    column = open_mutable_reason_column(group, chunk_size=2)

    assert column is not None
    assert "reason" not in group
    column[1:3] = np.asarray(["manual_correction", "geometry_issue"], dtype=object)
    assert read_reason_labels(group).tolist() == [
        "clean",
        "manual_correction",
        "geometry_issue",
    ]
