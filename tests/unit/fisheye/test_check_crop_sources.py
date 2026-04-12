from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fisheye.diagnostics import check_crop_sources as mod


class _FakeArray:
    def __init__(self, data: Any) -> None:
        self._data = np.asarray(data)
        self.shape = self._data.shape

    def __getitem__(self, item):
        return self._data[item]


class _FakeGroup:
    def __init__(self, *, path: str = "") -> None:
        self._children: dict[str, _FakeGroup | _FakeArray] = {}
        self.attrs: dict[str, Any] = {}
        self.path = path

    def create_group(self, name: str) -> "_FakeGroup":
        if "/" in name:
            head, tail = name.split("/", 1)
            return self.require_group(head).create_group(tail)
        child_path = f"{self.path}/{name}" if self.path else name
        child = _FakeGroup(path=child_path)
        self._children[name] = child
        return child

    def require_group(self, name: str) -> "_FakeGroup":
        if "/" in name:
            head, tail = name.split("/", 1)
            return self.require_group(head).require_group(tail)
        existing = self._children.get(name)
        if isinstance(existing, _FakeGroup):
            return existing
        child_path = f"{self.path}/{name}" if self.path else name
        child = _FakeGroup(path=child_path)
        self._children[name] = child
        return child

    def create_array(self, name: str, *, data: Any) -> _FakeArray:
        arr = _FakeArray(data)
        self._children[name] = arr
        return arr

    def get(self, name: str):
        try:
            return self[name]
        except Exception:
            return None

    def __contains__(self, key: str) -> bool:
        try:
            _ = self[key]
            return True
        except Exception:
            return False

    def __getitem__(self, key: str):
        if "/" in key:
            current: _FakeGroup | _FakeArray = self
            for token in key.split("/"):
                if not isinstance(current, _FakeGroup):
                    raise KeyError(key)
                current = current._children[token]
            return current
        return self._children[key]


@dataclass(frozen=True)
class _FakeResolution:
    detection_path: str
    refined_detect_run: str


def test_expected_refined_path_prefers_sparse_instances(monkeypatch) -> None:
    root = _FakeGroup()
    refined_parent = root.create_group("refined_detect_runs")
    refined_parent.attrs["latest"] = "refined_detect_001"
    refined_parent.create_group("refined_detect_001")

    monkeypatch.setattr(
        mod,
        "resolve_detection_read_source",
        lambda *_args, **_kwargs: _FakeResolution(
            detection_path="refined_detect_runs/refined_detect_001/instances",
            refined_detect_run="refined_detect_001",
        ),
    )

    parent_name, expected_path = mod._expected_refined_path(root)  # type: ignore[arg-type]

    assert parent_name == "refined_detect_runs"
    assert expected_path == "refined_detect_runs/refined_detect_001/instances"


def test_analyze_crop_run_flags_stale_sparse_root_alias() -> None:
    root = _FakeGroup()
    crop_parent = root.create_group("crop_runs")
    crop = crop_parent.create_group("crop_001")
    crop.attrs["detection_source_type"] = "refined"
    crop.attrs["detection_source_path"] = "refined_detect_runs/refined_detect_001"
    crop.create_array("frame_indices", data=np.array([0, 2], dtype=np.int32))

    refined_parent = root.create_group("refined_detect_runs")
    refined = refined_parent.create_group("refined_detect_001")
    instances = refined.create_group("instances")
    instances.create_array("frame_indices", data=np.array([0, 2], dtype=np.int32))

    row = mod.analyze_crop_run(
        root,  # type: ignore[arg-type]
        "crop_001",
        "refined_detect_runs/refined_detect_001/instances",
    )

    assert row[3] == "stale refined root"
    assert row[4] == "yes"
    assert "expected canonical sparse path refined_detect_runs/refined_detect_001/instances" in row[6]
