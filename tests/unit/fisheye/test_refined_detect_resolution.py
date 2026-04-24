from __future__ import annotations

from typing import Any

from fisheye.shared.refined_detect_resolution import (
    REVIEW_STATUS_DETECT_GROUP_PREFERENCE,
    resolve_detect_review_target,
    resolve_active_curated_refined_run_name,
    resolve_detection_read_source,
)
from fisheye.shared.refined_detect_review import (
    DEFAULT_DETECT_GROUP_PREFERENCE,
    resolve_refined_detect_group,
)


class _FakeGroup:
    def __init__(self, *, path: str = "") -> None:
        self._children: dict[str, _FakeGroup] = {}
        self.attrs: dict[str, Any] = {}
        self.path = path

    def create_group(self, name: str) -> "_FakeGroup":
        if "/" in name:
            head, tail = name.split("/", 1)
            return self.require_group(head).create_group(tail)
        if name in self._children:
            raise ValueError(f"{name} already exists")
        child_path = f"{self.path}/{name}" if self.path else name
        child = _FakeGroup(path=child_path)
        self._children[name] = child
        return child

    def require_group(self, name: str) -> "_FakeGroup":
        if "/" in name:
            head, tail = name.split("/", 1)
            return self.require_group(head).require_group(tail)
        existing = self._children.get(name)
        if existing is not None:
            return existing
        child_path = f"{self.path}/{name}" if self.path else name
        child = _FakeGroup(path=child_path)
        self._children[name] = child
        return child

    def get(self, name: str):
        try:
            return self[name]
        except Exception:
            return None

    def group_keys(self):
        return list(self._children.keys())

    def __contains__(self, key: str) -> bool:
        try:
            _ = self[key]
            return True
        except Exception:
            return False

    def __getitem__(self, key: str):
        if "/" in key:
            current: _FakeGroup = self
            for token in key.split("/"):
                current = current._children[token]
            return current
        return self._children[key]


def _seed_curated_run(run: _FakeGroup) -> None:
    for name in (
        "refined_row_ids",
        "frame_indices",
        "entity_ids",
        "bbox_img_xyxy",
        "bbox_norm_coords",
        "status_codes",
        "source_kind_codes",
        "source_sparse_row_index",
        "source_sparse_group_codes",
        "review_state_codes",
        "keypoints_state_codes",
        "subject_mask_state_codes",
        "eye_mask_state_codes",
        "swim_bladder_state_codes",
    ):
        run._children[name] = _FakeGroup(path=f"{run.path}/{name}")


def _seed_curated_instances(run: _FakeGroup) -> None:
    instances = run.create_group("instances")
    for name in (
        "refined_row_ids",
        "frame_indices",
        "frame_offsets",
        "bbox_img_xyxy",
        "bbox_norm_coords",
        "source_kind_codes",
        "manual_edit_flags",
        "source_detect_row_index",
        "frame_counts",
    ):
        instances._children[name] = _FakeGroup(path=f"{instances.path}/{name}")


def test_resolve_active_curated_refined_run_name_prefers_latest_with_root_arrays() -> None:
    root = _FakeGroup()
    parent = root.create_group("refined_detect_runs")
    parent.attrs["latest"] = "refined_detect_new"
    old = parent.create_group("refined_detect_old")
    new = parent.create_group("refined_detect_new")
    _seed_curated_run(old)
    _seed_curated_run(new)

    resolved = resolve_active_curated_refined_run_name(root)  # type: ignore[arg-type]

    assert resolved == "refined_detect_new"


def test_resolve_detection_read_source_prefers_curated_refined_root() -> None:
    root = _FakeGroup()
    refined_parent = root.create_group("refined_detect_runs")
    refined_parent.attrs["latest"] = "refined_detect_001"
    refined = refined_parent.create_group("refined_detect_001")
    refined.attrs["source_detect_run"] = "detect_001"
    refined.attrs["active_sparse_group"] = "manual_a"
    _seed_curated_run(refined)

    resolved = resolve_detection_read_source(root)  # type: ignore[arg-type]

    assert resolved.detection_kind == "refined"
    assert resolved.detection_path == "refined_detect_runs/refined_detect_001"
    assert resolved.refined_detect_run == "refined_detect_001"
    assert resolved.refined_sparse_group == "manual_a"
    assert resolved.source_detect_run == "detect_001"
    assert resolved.curated_root is True


def test_resolve_detection_read_source_falls_back_to_sparse_manual_when_curated_root_absent() -> None:
    root = _FakeGroup()
    refined_parent = root.create_group("refined_detect_runs")
    refined_parent.attrs["latest"] = "refined_detect_001"
    refined = refined_parent.create_group("refined_detect_001")
    refined.attrs["source_detect_run"] = "detect_001"
    refined.attrs["manual_review_latest"] = "manual_a"
    refined.create_group("manual_a")
    refined.create_group("interpolated")

    resolved = resolve_detection_read_source(root)  # type: ignore[arg-type]

    assert resolved.detection_kind == "manual"
    assert resolved.detection_path == "refined_detect_runs/refined_detect_001/manual_a"
    assert resolved.refined_detect_run == "refined_detect_001"
    assert resolved.refined_sparse_group == "manual_a"
    assert resolved.curated_root is False


def test_resolve_detection_read_source_accepts_instances_only_curated_surface() -> None:
    root = _FakeGroup()
    refined_parent = root.create_group("refined_detect_runs")
    refined_parent.attrs["latest"] = "refined_detect_001"
    refined = refined_parent.create_group("refined_detect_001")
    refined.attrs["source_detect_run"] = "detect_001"
    _seed_curated_instances(refined)

    resolved = resolve_detection_read_source(root)  # type: ignore[arg-type]

    assert resolved.detection_kind == "refined"
    assert resolved.detection_path == "refined_detect_runs/refined_detect_001/instances"
    assert resolved.refined_detect_run == "refined_detect_001"
    assert resolved.source_detect_run == "detect_001"
    assert resolved.curated_root is True


def test_resolve_refined_detect_group_prefers_instances_over_legacy_manual() -> None:
    refined = _FakeGroup(path="refined_detect_runs/refined_detect_001")
    refined.attrs["source_detect_run"] = "detect_001"
    refined.attrs["manual_review_latest"] = "manual_a"
    refined.create_group("manual_a")
    _seed_curated_instances(refined)

    resolved = resolve_refined_detect_group(  # type: ignore[arg-type]
        refined,
        preference=DEFAULT_DETECT_GROUP_PREFERENCE,
    )

    assert resolved.label == "refined"
    assert resolved.group == "instances"
    assert resolved.source_detect_run == "detect_001"


def test_resolve_detect_review_target_prefers_curated_refined_surface() -> None:
    root = _FakeGroup()
    refined_parent = root.create_group("refined_detect_runs")
    refined_parent.attrs["latest"] = "refined_detect_001"
    refined = refined_parent.create_group("refined_detect_001")
    refined.attrs["source_detect_run"] = "detect_001"
    _seed_curated_run(refined)

    resolved = resolve_detect_review_target(  # type: ignore[arg-type]
        root,
        refined_run_name="refined_detect_001",
        refined_run=refined,
    )

    assert resolved.resolved_group == "refined"
    assert tuple(resolved.preference_chain) == REVIEW_STATUS_DETECT_GROUP_PREFERENCE


def test_resolve_detect_review_target_falls_back_to_sparse_manual_when_curated_absent() -> None:
    root = _FakeGroup()
    refined_parent = root.create_group("refined_detect_runs")
    refined_parent.attrs["latest"] = "refined_detect_001"
    refined = refined_parent.create_group("refined_detect_001")
    refined.attrs["source_detect_run"] = "detect_001"
    refined.attrs["manual_review_latest"] = "manual_a"
    refined.create_group("manual_a")

    resolved = resolve_detect_review_target(  # type: ignore[arg-type]
        root,
        refined_run_name="refined_detect_001",
        refined_run=refined,
    )

    assert resolved.resolved_group == "manual"
    assert tuple(resolved.preference_chain) == REVIEW_STATUS_DETECT_GROUP_PREFERENCE


def test_resolve_detect_review_target_normalizes_override_group_case() -> None:
    root = _FakeGroup()
    refined_parent = root.create_group("refined_detect_runs")
    refined_parent.attrs["latest"] = "refined_detect_001"
    refined = refined_parent.create_group("refined_detect_001")
    refined.attrs["source_detect_run"] = "detect_001"
    refined.attrs["manual_review_latest"] = "manual_a"
    refined.create_group("manual_a")
    refined.create_group("interpolated")
    _seed_curated_run(refined)

    resolved_refined = resolve_detect_review_target(  # type: ignore[arg-type]
        root,
        refined_run_name="refined_detect_001",
        refined_run=refined,
        override_group="REFINED",
    )
    resolved_interpolated = resolve_detect_review_target(  # type: ignore[arg-type]
        root,
        refined_run_name="refined_detect_001",
        refined_run=refined,
        override_group="INTERPOLATED",
    )
    resolved_raw = resolve_detect_review_target(  # type: ignore[arg-type]
        root,
        refined_run_name="refined_detect_001",
        refined_run=refined,
        override_group="RAW",
    )

    assert resolved_refined.resolved_group == "refined"
    assert resolved_interpolated.resolved_group == "interpolated"
    assert resolved_raw.resolved_group == "raw"
