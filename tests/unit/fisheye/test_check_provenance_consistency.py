from __future__ import annotations

from typing import Any

from fisheye.diagnostics import check_provenance_consistency as mod


class _FakeArray:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class _FakeGroup:
    def __init__(self, path: str = "") -> None:
        self.path = path
        self.attrs: dict[str, Any] = {}
        self._children: dict[str, Any] = {}

    def create_group(self, name: str) -> "_FakeGroup":
        child_path = f"{self.path}/{name}" if self.path else name
        child = _FakeGroup(child_path)
        self._children[name] = child
        return child

    def create_array(self, name: str, shape: tuple[int, ...]) -> _FakeArray:
        arr = _FakeArray(shape)
        self._children[name] = arr
        return arr

    def _resolve(self, key: str) -> Any | None:
        node: Any = self
        parts = [p for p in key.split("/") if p]
        for part in parts:
            if not isinstance(node, _FakeGroup):
                return None
            node = node._children.get(part)
            if node is None:
                return None
        return node

    def get(self, key: str) -> Any | None:
        return self._resolve(key)

    def __contains__(self, key: str) -> bool:
        return self._resolve(key) is not None

    def __getitem__(self, key: str) -> Any:
        node = self._resolve(key)
        if node is None:
            raise KeyError(key)
        return node

    def group_keys(self):
        return [name for name, value in self._children.items() if isinstance(value, _FakeGroup)]

    def keys(self):
        return self._children.keys()


def _build_root_with_detect() -> _FakeGroup:
    root = _FakeGroup()
    detect_runs = root.create_group("detect_runs")
    detect = detect_runs.create_group("detect_001")
    detect.create_array("bbox_norm_coords", shape=(3, 4))
    detect_runs.attrs["latest"] = "detect_001"
    return root


def test_collect_provenance_handles_missing_refined_interpolated_bbox() -> None:
    root = _build_root_with_detect()

    refined_runs = root.create_group("refined_detect_runs")
    refined = refined_runs.create_group("refined_detect_001")
    refined.attrs["source_detect_run"] = "detect_001"
    refined.create_group("interpolated")  # Intentionally missing bbox_norm_coords.
    refined_runs.attrs["latest"] = "refined_detect_001"

    record = mod.collect_provenance(root)  # type: ignore[arg-type]

    assert record.detect_rows == 3
    assert record.refined_rows is None
    assert any("missing detection arrays" in issue for issue in record.issues)


def test_collect_provenance_handles_missing_optional_stage_arrays() -> None:
    root = _build_root_with_detect()

    crop_runs = root.create_group("crop_runs")
    crop = crop_runs.create_group("crop_001")
    crop.attrs["detection_source_path"] = "detect_runs/detect_001"
    crop_runs.attrs["latest"] = "crop_001"

    keypoints_runs = root.create_group("keypoints_runs")
    keypoints = keypoints_runs.create_group("keypoints_001")
    keypoints.attrs["source_crop_run"] = "crop_001"
    keypoints_runs.attrs["latest"] = "keypoints_001"

    arena_runs = root.create_group("arena_assignment_runs")
    arena_run = arena_runs.create_group("arena_001")
    arena_run.attrs["source_detect_run"] = "detect_001"
    arena_runs.attrs["latest"] = "arena_001"

    record = mod.collect_provenance(root)  # type: ignore[arg-type]

    assert record.crop_rois is None
    assert record.keypoint_rows is None
    assert record.arena_assignment_rows is None
