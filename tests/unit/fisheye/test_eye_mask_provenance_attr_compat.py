from __future__ import annotations

from typing import Any

import numpy as np

from fisheye.diagnostics import check_eye_masks, check_full_provenance
from fisheye.shared.provenance_attrs import (
    build_source_keypoints_attrs,
    resolve_source_keypoints_run,
)


class _FakeArray:
    def __init__(self, data: np.ndarray) -> None:
        self._data = data

    def __getitem__(self, item: Any) -> np.ndarray:
        return self._data[item]


class _FakeGroup(dict):
    def __init__(self, *args: Any, attrs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.attrs = attrs or {}

    def get(self, key: str, default: Any = None) -> Any:
        return super().get(key, default)


def test_resolve_source_keypoints_run_prefers_canonical() -> None:
    attrs = {
        "source_keypoints_run": "kp_canonical",
        "source_keypoint_run": "kp_legacy",
    }
    assert resolve_source_keypoints_run(attrs) == "kp_canonical"


def test_resolve_source_keypoints_run_falls_back_to_legacy() -> None:
    attrs = {"source_keypoint_run": "kp_legacy_only"}
    assert resolve_source_keypoints_run(attrs) == "kp_legacy_only"


def test_build_source_keypoints_attrs_writes_canonical_and_legacy() -> None:
    payload = build_source_keypoints_attrs("kp_001", include_legacy_alias=True)
    assert payload["source_keypoints_run"] == "kp_001"
    assert payload["source_keypoint_run"] == "kp_001"


def test_check_keypoint_lineage_attrs_reports_legacy_only() -> None:
    details: list[str] = []
    status = check_eye_masks._check_keypoint_lineage_attrs(
        {"source_keypoint_run": "kp_legacy_only"},
        current_status="[green]healthy[/green]",
        details=details,
    )
    assert status == "[yellow]legacy[/yellow]"
    assert any("legacy attr 'source_keypoint_run'" in line for line in details)


def test_check_keypoint_lineage_attrs_reports_missing_attrs() -> None:
    details: list[str] = []
    status = check_eye_masks._check_keypoint_lineage_attrs(
        {},
        current_status="[green]healthy[/green]",
        details=details,
    )
    assert status == "[yellow]incomplete[/yellow]"
    assert any("missing attr 'source_keypoints_run'" in line for line in details)


def test_load_eye_mask_provenance_prefers_canonical_attr() -> None:
    eye_run = _FakeGroup(
        attrs={
            "source_detect_run": "detect_001",
            "source_keypoints_run": "kp_canonical",
            "source_keypoint_run": "kp_legacy",
        }
    )
    eye_run["detection_source"] = _FakeArray(np.array([0, 1], dtype=np.int8))

    eye_parent = _FakeGroup(attrs={"latest": "eye_001"})
    eye_parent["eye_001"] = eye_run

    root = _FakeGroup({"eye_masks_runs": eye_parent})
    run, detect_run, keypoint_run, arr = check_full_provenance._load_eye_mask_provenance(root)
    assert run == "eye_001"
    assert detect_run == "detect_001"
    assert keypoint_run == "kp_canonical"
    assert np.array_equal(arr, np.array([0, 1], dtype=np.int8))


def test_load_eye_mask_provenance_falls_back_to_legacy_attr() -> None:
    eye_run = _FakeGroup(attrs={"source_detect_run": "detect_001", "source_keypoint_run": "kp_legacy_only"})
    eye_run["detection_source"] = _FakeArray(np.array([0, 0], dtype=np.int8))

    eye_parent = _FakeGroup(attrs={"latest": "eye_001"})
    eye_parent["eye_001"] = eye_run

    root = _FakeGroup({"eye_masks_runs": eye_parent})
    run, detect_run, keypoint_run, arr = check_full_provenance._load_eye_mask_provenance(root)
    assert run == "eye_001"
    assert detect_run == "detect_001"
    assert keypoint_run == "kp_legacy_only"
    assert np.array_equal(arr, np.array([0, 0], dtype=np.int8))

