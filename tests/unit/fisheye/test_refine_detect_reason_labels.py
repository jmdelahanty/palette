import numpy as np
import zarr

from fisheye.shared.detect_reason_codec import decode_reason_bytes, read_reason_labels
from fisheye.refinement.refine_detect import (
    _build_filtered_reason_labels,
    _build_interpolated_reason_labels,
    _write_reason_array,
    get_refinement_parameters,
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
    reason_bytes = np.asarray(grp["reason_bytes"][:], dtype=np.uint8)
    decoded = decode_reason_bytes(reason_bytes).tolist()
    assert decoded == ["clean", "interpolated", "clean"]
    assert grp.attrs["reason_encoding"] == "utf8-null-terminated"
    assert grp.attrs["reason_fallback_order"] == ["reason_bytes", "reason", "detection_source"]


def test_read_reason_labels_falls_back_to_reason_bytes(tmp_path) -> None:
    root = zarr.open_group(store=tmp_path / "test_reason_bytes.zarr", mode="w")
    grp = root.create_group("filtered")
    _write_reason_array(grp, np.array(["clean", "manual"], dtype=object), chunk_size=2)
    del grp["reason"]

    labels = read_reason_labels(grp)
    assert labels is not None
    assert labels.tolist() == ["clean", "manual"]


def test_read_reason_labels_falls_back_to_detection_source(tmp_path) -> None:
    root = zarr.open_group(store=tmp_path / "test_source_fallback.zarr", mode="w")
    grp = root.create_group("interpolated")
    grp.create_array("detection_source", data=np.array([0, 1, 0, 1], dtype=np.int8))

    labels = read_reason_labels(grp)
    assert labels is not None
    assert labels.tolist() == ["clean", "interpolated", "clean", "interpolated"]


def test_get_refinement_parameters_defaults_max_gap_to_50() -> None:
    params, source = get_refinement_parameters(config={})
    assert source == "config"
    assert params["max_gap"] == 50
