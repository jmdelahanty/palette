from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import zarr
from zarr.core.dtype import VariableLengthUTF8
from zarr.errors import UnstableSpecificationWarning

from fisheye.utils.audit_zarr_string_encodings import audit_archive


def test_audit_archive_counts_string_encodings(tmp_path: Path) -> None:
    zarr_path = tmp_path / "encoding_audit.zarr"
    root = zarr.open_group(store=zarr_path, mode="w")
    root.attrs["zarr_purpose"] = "analysis"

    root.create_array("reason_bytes", data=np.zeros((3, 16), dtype=np.uint8), chunks=(3, 16))

    reason = root.create_array(
        "reason",
        shape=(3,),
        dtype=VariableLengthUTF8(),
        fill_value="",
        chunks=(3,),
    )
    reason[:] = np.asarray(["clean", "interpolated", "manual"], dtype=object)

    # These legacy fixed-width encodings intentionally trigger Zarr v3 unstable-spec warnings.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UnstableSpecificationWarning)
        root.create_array("source_dataset_id", data=np.array(["dataset_a", "dataset_b"], dtype="<U16"), chunks=(2,))
        root.create_array("source_dataset_path", data=np.array([b"/a", b"/b"], dtype="S8"), chunks=(2,))
    warning_texts = [str(item.message) for item in caught]
    assert any("FixedLengthUTF32" in text for text in warning_texts)
    assert any("NullTerminatedBytes" in text for text in warning_texts)

    report = audit_archive(zarr_path, zarr_use_filter="any")
    counts = report["counts"]

    assert report["filtered_zarr_use"] is False
    assert counts["reason_bytes"] == 1
    assert counts["vlen_utf8"] == 1
    assert counts["fixed_unicode"] == 1
    assert counts["fixed_bytes"] == 1
    assert counts["object"] == 0


def test_audit_archive_respects_zarr_use_filter(tmp_path: Path) -> None:
    zarr_path = tmp_path / "encoding_filter.zarr"
    root = zarr.open_group(store=zarr_path, mode="w")
    root.attrs["zarr_purpose"] = "analysis"
    root.create_array("reason_bytes", data=np.zeros((1, 8), dtype=np.uint8), chunks=(1, 8))

    report = audit_archive(zarr_path, zarr_use_filter="training")

    assert report["filtered_zarr_use"] is True
    assert report["arrays_scanned"] == 0
    assert report["counts"] == {}
