import numpy as np
import zarr

from fisheye.refinement.refine_detect import (
    _build_filtered_reason_labels,
    _build_interpolated_reason_labels,
    _write_reason_array,
)


def test_build_filtered_reason_labels_all_clean() -> None:
    labels = _build_filtered_reason_labels(4)
    assert labels.dtype == object
    assert labels.tolist() == ["clean", "clean", "clean", "clean"]


def test_build_interpolated_reason_labels_maps_source() -> None:
    source = np.array([0, 1, 1, 0, 0], dtype=np.int8)
    labels = _build_interpolated_reason_labels(source)
    assert labels.dtype == object
    assert labels.tolist() == ["clean", "interpolated", "interpolated", "clean", "clean"]


def test_write_reason_array_round_trip(tmp_path) -> None:
    root = zarr.open_group(store=tmp_path / "test.zarr", mode="w")
    grp = root.create_group("interpolated")
    reason = np.array(["clean", "interpolated", "clean"], dtype=object)

    _write_reason_array(grp, reason, chunk_size=2)

    stored = np.asarray(grp["reason"][:], dtype=object).tolist()
    assert stored == ["clean", "interpolated", "clean"]
