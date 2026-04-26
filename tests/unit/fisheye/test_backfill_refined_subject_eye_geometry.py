from __future__ import annotations

from typing import Any

import numpy as np

import fisheye.utils.backfill_refined_subject_eye_geometry as mod


class _FakeArray:
    def __init__(self, data: Any) -> None:
        self._data = np.asarray(data)

    def __getitem__(self, key):
        return self._data[key]

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape


class _FakeGroup(dict):
    def __init__(self, *args, attrs: dict[str, Any] | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attrs = attrs or {}

    def _resolve_parent(self, path: str) -> tuple["_FakeGroup", str]:
        tokens = [token for token in path.split("/") if token]
        if not tokens:
            raise KeyError(path)
        parent: _FakeGroup = self
        for token in tokens[:-1]:
            child = parent.get(token)
            if not isinstance(child, _FakeGroup):
                child = _FakeGroup()
                parent[token] = child
            parent = child
        return parent, tokens[-1]

    def require_group(self, path: str) -> "_FakeGroup":
        parent, name = self._resolve_parent(path)
        child = parent.get(name)
        if isinstance(child, _FakeGroup):
            return child
        child = _FakeGroup()
        parent[name] = child
        return child

    def create_array(self, name: str, *, data: Any, **_kwargs) -> _FakeArray:
        parent, leaf = self._resolve_parent(name)
        array = _FakeArray(data)
        parent[leaf] = array
        return array

    def group_keys(self) -> list[str]:
        return [key for key, value in self.items() if isinstance(value, _FakeGroup)]

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __getitem__(self, key: str):
        tokens = [token for token in key.split("/") if token]
        if not tokens:
            raise KeyError(key)
        current: Any = self
        for token in tokens:
            if not isinstance(current, _FakeGroup):
                raise KeyError(key)
            current = dict.__getitem__(current, token)
        return current

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        try:
            self[key]
            return True
        except KeyError:
            return False


def _run_group(*, available: list[bool] | None = None, labels: list[str] | None = None) -> _FakeGroup:
    labels = labels if labels is not None else ["subject_body", "eye_left", "eye_right", "swim_bladder"]
    group = _FakeGroup(attrs={"mask_labels": labels})
    group.create_array("masks_roi", data=np.zeros((3, len(labels), 8, 8), dtype=np.uint8))
    if available is not None:
        group.create_array("available_channels", data=np.asarray(available, dtype=bool))
    return group


def _add_existing_geometry(group: _FakeGroup) -> None:
    for component in mod.EYE_COMPONENTS:
        geometry = group.require_group(f"components/{component}/geometry")
        geometry.create_array("ellipse_params", data=np.zeros((3, 5), dtype=np.float32))
        geometry.create_array("ellipse_success", data=np.zeros((3,), dtype=bool))
        contours = group.require_group(f"components/{component}/contours")
        contours.create_array("ptr", data=np.zeros((3,), dtype=np.int64))
        contours.create_array("len", data=np.zeros((3,), dtype=np.int32))
        contours.create_array("points_xy", data=np.zeros((1, 2), dtype=np.float32))
    metrics = group.require_group("relations/eye_pair/metrics")
    metrics.create_array("separation_px", data=np.zeros((3,), dtype=np.float32))
    metrics.create_array("separation_valid", data=np.zeros((3,), dtype=bool))


def test_dry_run_marks_eligible_lr_eye_run_ok_without_writing(monkeypatch) -> None:
    group = _run_group(available=[True, True, True, True])
    calls: list[_FakeGroup] = []
    monkeypatch.setattr(mod, "write_refined_subject_eye_geometry", lambda run_group: calls.append(run_group))

    result = mod._backfill_run_group(group, apply=False)

    assert result.status == "ok"
    assert result.roi_count == 3
    assert result.geometry_existing is False
    assert calls == []


def test_apply_reuses_writer_and_reports_existing_refresh(monkeypatch) -> None:
    group = _run_group(available=[True, True, True, True])
    _add_existing_geometry(group)
    calls: list[_FakeGroup] = []

    def fake_writer(run_group: _FakeGroup) -> dict[str, object]:
        calls.append(run_group)
        return {
            "status": "updated",
            "roi_count": 3,
            "ellipse_success_count": 6,
            "pair_success_count": 3,
        }

    monkeypatch.setattr(mod, "write_refined_subject_eye_geometry", fake_writer)

    result = mod._backfill_run_group(group, apply=True)

    assert calls == [group]
    assert result.status == "ok"
    assert result.geometry_existing is True
    assert result.ellipse_success_count == 6
    assert result.pair_success_count == 3


def test_writer_marks_eye_geometry_computed_and_clears_deferred_status() -> None:
    group = _run_group(available=[True, True, True, True])
    group.attrs["eye_geometry_status"] = "deferred"
    group.attrs["eye_geometry_deferred_reason"] = "write_eye_geometry=false"

    result = mod.write_refined_subject_eye_geometry(group)

    assert result["status"] == "updated"
    assert group.attrs["eye_geometry_status"] == "computed"
    assert "eye_geometry_deferred_reason" not in group.attrs
    assert "relations/eye_pair/metrics/separation_valid" in group


def test_missing_eye_labels_are_not_eligible() -> None:
    group = _run_group(labels=["subject_body", "eye", "swim_bladder"])

    result = mod._backfill_run_group(group, apply=False)

    assert result.status == "no_lr_eyes"
    assert "missing eye_left/eye_right" in str(result.reason)


def test_unavailable_eye_channels_are_not_recomputed() -> None:
    group = _run_group(available=[True, True, False, True])

    result = mod._backfill_run_group(group, apply=False)

    assert result.status == "unavailable_eyes"
    assert result.roi_count == 3
    assert "eye_right" in str(result.reason)


def test_latest_run_resolution_prefers_parent_latest() -> None:
    root = _FakeGroup()
    parent = root.require_group("refined_subject_masks_runs")
    parent.attrs["latest"] = "run_b"
    parent.require_group("run_a")
    parent.require_group("run_b")

    run_info = list(mod._iter_run_groups(root, all_runs=False))

    assert [run_path for run_path, _run_group in run_info] == ["refined_subject_masks_runs/run_b"]
