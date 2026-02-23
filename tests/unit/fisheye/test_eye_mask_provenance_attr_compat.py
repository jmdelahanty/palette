from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from fisheye.diagnostics import check_eye_masks, check_full_provenance
from fisheye.segmentation.eye_segmentation import _resolve_keypoint_group as _resolve_traditional_keypoint_group
from fisheye.segmentation.eye_segmentation_yolo import _resolve_keypoint_lineage as _resolve_yolo_keypoint_lineage
from fisheye.segmentation.infer_unet_eye_masks import (
    _resolve_source_keypoints_run_for_unet,
)
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


def _build_keypoint_root() -> _FakeGroup:
    refined = _FakeGroup(attrs={"latest": "kp_refined_latest"})
    refined["kp_refined_latest"] = _FakeGroup()
    refined["kp_shared"] = _FakeGroup()

    raw = _FakeGroup(attrs={"latest": "kp_raw_latest"})
    raw["kp_raw_latest"] = _FakeGroup()
    raw["kp_raw_only"] = _FakeGroup()
    raw["kp_shared"] = _FakeGroup()

    return _FakeGroup(
        {
            "refined_keypoints_runs": refined,
            "keypoints_runs": raw,
        }
    )


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


def test_traditional_resolve_keypoint_group_prefers_explicit_run() -> None:
    root = _build_keypoint_root()
    group, run_name, group_name = _resolve_traditional_keypoint_group(root, keypoint_run="kp_raw_only")
    assert run_name == "kp_raw_only"
    assert group_name == "keypoints_runs"
    assert group is root["keypoints_runs"]["kp_raw_only"]


def test_traditional_resolve_keypoint_group_defaults_to_refined_latest() -> None:
    root = _build_keypoint_root()
    group, run_name, group_name = _resolve_traditional_keypoint_group(root, keypoint_run=None)
    assert run_name == "kp_refined_latest"
    assert group_name == "refined_keypoints_runs"
    assert group is root["refined_keypoints_runs"]["kp_refined_latest"]


def test_yolo_resolve_keypoint_lineage_prefers_canonical_attr() -> None:
    root = _build_keypoint_root()
    crop_group = _FakeGroup(
        attrs={
            "source_keypoints_run": "kp_raw_only",
            "source_keypoint_run": "kp_shared",
        }
    )
    run_name, group_name = _resolve_yolo_keypoint_lineage(root, crop_group, keypoints_run=None)
    assert run_name == "kp_raw_only"
    assert group_name == "keypoints_runs"


def test_yolo_resolve_keypoint_lineage_falls_back_to_legacy_attr() -> None:
    root = _build_keypoint_root()
    crop_group = _FakeGroup(attrs={"source_keypoint_run": "kp_raw_only"})
    run_name, group_name = _resolve_yolo_keypoint_lineage(root, crop_group, keypoints_run=None)
    assert run_name == "kp_raw_only"
    assert group_name == "keypoints_runs"


def test_unet_source_keypoint_resolution_prefers_explicit_override() -> None:
    resolved = _resolve_source_keypoints_run_for_unet(
        explicit_keypoints_run="kp_override",
        source_attrs={"source_keypoints_run": "kp_src"},
        latest_keypoints_run="kp_latest",
    )
    assert resolved == "kp_override"


def test_unet_source_keypoint_resolution_prefers_canonical_over_legacy() -> None:
    resolved = _resolve_source_keypoints_run_for_unet(
        explicit_keypoints_run=None,
        source_attrs={
            "source_keypoints_run": "kp_canonical",
            "source_keypoint_run": "kp_legacy",
        },
        latest_keypoints_run="kp_latest",
    )
    assert resolved == "kp_canonical"


def test_unet_source_keypoint_resolution_falls_back_to_legacy_alias() -> None:
    resolved = _resolve_source_keypoints_run_for_unet(
        explicit_keypoints_run=None,
        source_attrs={"source_keypoint_run": "kp_legacy"},
        latest_keypoints_run="kp_latest",
    )
    assert resolved == "kp_legacy"


def test_unet_source_keypoint_resolution_falls_back_to_latest() -> None:
    resolved = _resolve_source_keypoints_run_for_unet(
        explicit_keypoints_run=None,
        source_attrs={},
        latest_keypoints_run="kp_latest",
    )
    assert resolved == "kp_latest"


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


@pytest.mark.parametrize(
    ("attrs", "expected_status", "expected_run"),
    [
        (
            {"source_keypoints_run": "kp_canonical", "source_keypoint_run": "kp_legacy"},
            "[green]healthy[/green]",
            "kp_canonical",
        ),
        (
            {"source_keypoint_run": "kp_legacy_only"},
            "[yellow]legacy[/yellow]",
            "kp_legacy_only",
        ),
        (
            {"source_keypoints_run": None, "source_keypoint_run": "kp_legacy_only"},
            "[green]healthy[/green]",
            "kp_legacy_only",
        ),
        (
            {"source_keypoints_run": None},
            "[yellow]incomplete[/yellow]",
            None,
        ),
        (
            {},
            "[yellow]incomplete[/yellow]",
            None,
        ),
    ],
)
def test_diagnostics_tools_agree_on_lineage_resolution_rules(
    attrs: dict[str, object],
    expected_status: str,
    expected_run: str | None,
) -> None:
    details: list[str] = []
    status = check_eye_masks._check_keypoint_lineage_attrs(
        attrs,
        current_status="[green]healthy[/green]",
        details=details,
    )
    assert status == expected_status

    eye_run = _FakeGroup(attrs={"source_detect_run": "detect_001", **attrs})
    eye_run["detection_source"] = _FakeArray(np.array([0, 1], dtype=np.int8))
    eye_parent = _FakeGroup(attrs={"latest": "eye_001"})
    eye_parent["eye_001"] = eye_run
    root = _FakeGroup({"eye_masks_runs": eye_parent})

    run, detect_run, keypoint_run, arr = check_full_provenance._load_eye_mask_provenance(root)
    assert run == "eye_001"
    assert detect_run == "detect_001"
    assert keypoint_run == expected_run
    assert np.array_equal(arr, np.array([0, 1], dtype=np.int8))
