from __future__ import annotations

from typing import Any

import pytest

from fisheye.refinement.refine_eye_masks import _resolve_keypoint_group
from fisheye.shared.provenance_attrs import resolve_source_keypoints_run


class _FakeGroup(dict):
    def __init__(self, *args: Any, attrs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.attrs = attrs or {}

    def get(self, key: str, default: Any = None) -> Any:
        return super().get(key, default)


def _build_root() -> _FakeGroup:
    refined = _FakeGroup(attrs={"latest": "kp_refined_latest"})
    refined["kp_refined_latest"] = _FakeGroup()
    refined["kp_refined_only"] = _FakeGroup()
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


def test_resolve_keypoint_group_uses_explicit_run() -> None:
    root = _build_root()
    grp, run_name, group_name = _resolve_keypoint_group(
        root,
        keypoint_run="kp_raw_only",
        source_keypoint_group=None,
        source_keypoint_run=None,
    )
    assert run_name == "kp_raw_only"
    assert group_name == "keypoints_runs"
    assert grp is root["keypoints_runs"]["kp_raw_only"]


def test_resolve_keypoint_group_fails_when_source_lineage_missing_by_default() -> None:
    root = _build_root()
    with pytest.raises(ValueError, match="Missing source keypoint lineage attrs"):
        _resolve_keypoint_group(
            root,
            keypoint_run=None,
            source_keypoint_group=None,
            source_keypoint_run=None,
        )


def test_resolve_keypoint_group_allows_latest_fallback_when_explicitly_enabled() -> None:
    root = _build_root()
    grp, run_name, group_name = _resolve_keypoint_group(
        root,
        keypoint_run=None,
        source_keypoint_group=None,
        source_keypoint_run=None,
        allow_latest_fallback=True,
    )
    assert run_name == "kp_refined_latest"
    assert group_name == "refined_keypoints_runs"
    assert grp is root["refined_keypoints_runs"]["kp_refined_latest"]


def test_resolve_keypoint_group_uses_source_run_without_group_when_unique() -> None:
    root = _build_root()
    grp, run_name, group_name = _resolve_keypoint_group(
        root,
        keypoint_run=None,
        source_keypoint_group=None,
        source_keypoint_run="kp_raw_only",
    )
    assert run_name == "kp_raw_only"
    assert group_name == "keypoints_runs"
    assert grp is root["keypoints_runs"]["kp_raw_only"]


def test_resolve_keypoint_group_rejects_ambiguous_source_run_without_group() -> None:
    root = _build_root()
    with pytest.raises(ValueError, match="exists in both refined and raw groups"):
        _resolve_keypoint_group(
            root,
            keypoint_run=None,
            source_keypoint_group=None,
            source_keypoint_run="kp_shared",
        )


def test_resolve_keypoint_group_rejects_group_without_run() -> None:
    root = _build_root()
    with pytest.raises(ValueError, match="group 'refined_keypoints_runs' is present but source keypoint run is missing"):
        _resolve_keypoint_group(
            root,
            keypoint_run=None,
            source_keypoint_group="refined_keypoints_runs",
            source_keypoint_run=None,
        )


def test_resolve_keypoint_group_accepts_legacy_source_attr_name() -> None:
    root = _build_root()
    source_attrs = {
        "source_keypoint_group": "keypoints_runs",
        "source_keypoint_run": "kp_raw_only",
    }
    grp, run_name, group_name = _resolve_keypoint_group(
        root,
        keypoint_run=None,
        source_keypoint_group=source_attrs["source_keypoint_group"],
        source_keypoint_run=resolve_source_keypoints_run(source_attrs),
    )
    assert run_name == "kp_raw_only"
    assert group_name == "keypoints_runs"
    assert grp is root["keypoints_runs"]["kp_raw_only"]


def test_resolve_keypoint_group_prefers_canonical_source_attr_when_both_present() -> None:
    root = _build_root()
    source_attrs = {
        "source_keypoints_run": "kp_raw_only",
        "source_keypoint_run": "kp_shared",
    }
    grp, run_name, group_name = _resolve_keypoint_group(
        root,
        keypoint_run=None,
        source_keypoint_group=None,
        source_keypoint_run=resolve_source_keypoints_run(source_attrs),
    )
    assert run_name == "kp_raw_only"
    assert group_name == "keypoints_runs"
    assert grp is root["keypoints_runs"]["kp_raw_only"]
