import numpy as np
import pytest

from fisheye.shared.detect_reason_codec import decode_reason_bytes, read_reason_labels
from fisheye.refinement.refine_detect import (
    _build_filtered_reason_labels,
    _build_interpolated_reason_labels,
    _resolve_detection_quality_labels,
    _write_reason_array,
    get_refinement_parameters,
)


class _FakeArray:
    def __init__(self, data: np.ndarray) -> None:
        self._data = data
        self.shape = data.shape

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, item, value) -> None:
        self._data[item] = value


class _FakeGroup:
    def __init__(self) -> None:
        self.attrs: dict[str, object] = {}
        self._children: dict[str, object] = {}

    def create_group(self, name: str):
        grp = _FakeGroup()
        self._children[name] = grp
        return grp

    def create_array(
        self,
        name: str,
        *,
        data=None,
        shape=None,
        fill_value=None,
        overwrite: bool = False,
        **_kwargs,
    ):
        if not overwrite and name in self._children:
            raise ValueError(f"Array '{name}' already exists")
        if data is None:
            if shape is None:
                raise ValueError("shape is required when data is None")
            arr = np.full(shape, fill_value if fill_value is not None else "", dtype=object)
        else:
            arr = np.asarray(data).copy()
        wrapped = _FakeArray(arr)
        self._children[name] = wrapped
        return wrapped

    def get(self, name: str):
        return self._children.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._children

    def __getitem__(self, name: str):
        return self._children[name]

    def __delitem__(self, name: str) -> None:
        del self._children[name]


def _fake_root() -> _FakeGroup:
    return _FakeGroup()


def test_build_filtered_reason_labels_all_clean() -> None:
    labels = _build_filtered_reason_labels(4)
    assert labels.dtype == object
    assert labels.tolist() == ["clean", "clean", "clean", "clean"]


def test_build_interpolated_reason_labels_maps_source() -> None:
    source = np.array([0, 1, 1, 0, 0], dtype=np.int8)
    labels = _build_interpolated_reason_labels(source)
    assert labels.dtype == object
    assert labels.tolist() == ["clean", "interpolated", "interpolated", "clean", "clean"]


def test_write_reason_array_round_trip() -> None:
    root = _fake_root()
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


def test_read_reason_labels_falls_back_to_reason_bytes() -> None:
    root = _fake_root()
    grp = root.create_group("filtered")
    _write_reason_array(grp, np.array(["clean", "manual"], dtype=object), chunk_size=2)
    del grp["reason"]

    labels = read_reason_labels(grp)
    assert labels is not None
    assert labels.tolist() == ["clean", "manual"]


def test_read_reason_labels_falls_back_to_detection_source() -> None:
    root = _fake_root()
    grp = root.create_group("interpolated")
    grp.create_array("detection_source", data=np.array([0, 1, 0, 1], dtype=np.int8))

    labels = read_reason_labels(grp)
    assert labels is not None
    assert labels.tolist() == ["clean", "interpolated", "clean", "interpolated"]


def test_get_refinement_parameters_defaults_max_gap_to_50() -> None:
    params, source = get_refinement_parameters(config={})
    assert source == "config"
    assert params["max_gap"] == 50


def test_resolve_detection_quality_labels_requires_quality_by_default() -> None:
    root = _fake_root()
    detect_group = root.create_group("detect_runs").create_group("detect_001")

    with pytest.raises(ValueError, match="Missing usable detect_quality context"):
        _resolve_detection_quality_labels(
            detect_group,
            detect_run="detect_001",
            quality_run=None,
            total_detections=3,
            require_quality=True,
            allow_missing_reason="test",
            console=None,
        )


def test_resolve_detection_quality_labels_allows_explicit_opt_out() -> None:
    root = _fake_root()
    detect_group = root.create_group("detect_runs").create_group("detect_001")

    labels, resolved_run, quality_group = _resolve_detection_quality_labels(
        detect_group,
        detect_run="detect_001",
        quality_run=None,
        total_detections=4,
        require_quality=False,
        allow_missing_reason="explicit opt-out",
        console=None,
    )

    assert labels.tolist() == [0, 0, 0, 0]
    assert resolved_run is None
    assert quality_group is None


def test_resolve_detection_quality_labels_uses_latest_quality_run() -> None:
    root = _fake_root()
    detect_group = root.create_group("detect_runs").create_group("detect_001")
    quality_reports = detect_group.create_group("quality_reports")
    quality_reports.attrs["latest"] = "detect_quality_001"
    quality_group = quality_reports.create_group("detect_quality_001")
    quality_group.create_array("detection_quality_labels", data=np.array([0, 2, 0], dtype=np.int8))

    labels, resolved_run, resolved_group = _resolve_detection_quality_labels(
        detect_group,
        detect_run="detect_001",
        quality_run=None,
        total_detections=3,
        require_quality=True,
        allow_missing_reason="test",
        console=None,
    )

    assert labels.tolist() == [0, 2, 0]
    assert resolved_run == "detect_quality_001"
    assert resolved_group is quality_group


def test_resolve_detection_quality_labels_rejects_length_mismatch() -> None:
    root = _fake_root()
    detect_group = root.create_group("detect_runs").create_group("detect_001")
    quality_reports = detect_group.create_group("quality_reports")
    quality_reports.attrs["latest"] = "detect_quality_001"
    quality_group = quality_reports.create_group("detect_quality_001")
    quality_group.create_array("detection_quality_labels", data=np.array([0, 2], dtype=np.int8))

    with pytest.raises(ValueError, match="does not match detections"):
        _resolve_detection_quality_labels(
            detect_group,
            detect_run="detect_001",
            quality_run=None,
            total_detections=3,
            require_quality=True,
            allow_missing_reason="test",
            console=None,
        )
