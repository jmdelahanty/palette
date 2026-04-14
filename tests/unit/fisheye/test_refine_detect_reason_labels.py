import numpy as np
import pytest

from fisheye.refinement.refine_detect import (
    _reject_deprecated_interpolation_overrides,
    _resolve_detection_quality_labels,
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


def test_get_refinement_parameters_defaults_max_gap_to_0() -> None:
    params, source = get_refinement_parameters(config={})
    assert source == "config"
    assert params["max_gap"] == 0
    assert params["interpolation_method"] == "disabled"


def test_reject_deprecated_interpolation_overrides() -> None:
    with pytest.raises(ValueError, match="Interpolation overrides are deprecated and unsupported"):
        _reject_deprecated_interpolation_overrides(max_gap=5, interpolation_method=None)


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
