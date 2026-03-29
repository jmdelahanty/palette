from __future__ import annotations

from pathlib import Path
from typing import Any

from fisheye.utils import batch_extend_keypoint_skeleton as mod


class _FakeGroup:
    def __init__(self, children: dict[str, Any] | None = None) -> None:
        self._children: dict[str, Any] = children or {}
        self.attrs: dict[str, Any] = {}

    def create_group(self, name: str) -> "_FakeGroup":
        child = _FakeGroup()
        self._children[name] = child
        return child

    def get(self, name: str):
        return self._children.get(name)

    def group_keys(self):
        return [key for key, value in self._children.items() if isinstance(value, _FakeGroup)]

    def keys(self):
        return self._children.keys()

    def __contains__(self, key: str) -> bool:
        return key in self._children

    def __getitem__(self, key: str):
        if "/" in key:
            current = self
            for token in key.split("/"):
                current = current._children[token]
            return current
        return self._children[key]


def _patch_scan(monkeypatch, mapping: dict[Path, _FakeGroup]) -> None:
    ordered_paths = list(mapping.keys())
    monkeypatch.setattr(mod, "_iter_zarr", lambda *_args, **_kwargs: iter(ordered_paths))
    monkeypatch.setattr(
        mod.zarr,
        "open_group",
        lambda path, mode="r": mapping[Path(path)],  # noqa: ARG005
    )


def test_main_uses_latest_refined_run_by_default(monkeypatch, tmp_path: Path, capsys) -> None:
    zarr_path = tmp_path / "recording_training.zarr"
    root = _FakeGroup()
    root.attrs["zarr_purpose"] = "training"
    refined_parent = root.create_group("refined_keypoints_runs")
    refined_parent.attrs["latest"] = "refined_latest"
    refined_parent.create_group("refined_old")
    refined_parent.create_group("refined_latest")

    seen: list[tuple[str, str]] = []

    def _fake_extend(root_arg, **kwargs):  # noqa: ANN003
        assert root_arg is root
        seen.append((kwargs["source_parent"], kwargs["source_run"]))
        return {
            "source_parent": kwargs["source_parent"],
            "source_run": kwargs["source_run"],
            "target_parent": kwargs["source_parent"],
            "target_run": kwargs["target_run"],
        }

    _patch_scan(monkeypatch, {zarr_path: root})
    monkeypatch.setattr(mod, "extend_keypoint_skeleton_run", _fake_extend)

    rc = mod.main([str(zarr_path), "--zarr-use", "training"])
    assert rc == 0
    assert seen == [("refined_keypoints_runs", "refined_latest")]
    out = capsys.readouterr().out
    assert "planned=1" in out


def test_main_skips_existing_target_without_overwrite(monkeypatch, tmp_path: Path, capsys) -> None:
    zarr_path = tmp_path / "recording_training.zarr"
    root = _FakeGroup()
    root.attrs["zarr_purpose"] = "training"
    refined_parent = root.create_group("refined_keypoints_runs")
    refined_parent.attrs["latest"] = "refined_latest"
    refined_parent.create_group("refined_latest")
    refined_parent.create_group("refined_latest_traditional_v2_seed")

    _patch_scan(monkeypatch, {zarr_path: root})

    rc = mod.main([str(zarr_path), "--zarr-use", "training"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "skipped_existing" in out
    assert "skipped_existing=1" in out


def test_main_skips_when_selected_source_already_uses_target_schema(monkeypatch, tmp_path: Path, capsys) -> None:
    zarr_path = tmp_path / "recording_training.zarr"
    root = _FakeGroup()
    root.attrs["zarr_purpose"] = "training"
    refined_parent = root.create_group("refined_keypoints_runs")
    refined_parent.attrs["latest"] = "refined_v2"
    refined_v2 = refined_parent.create_group("refined_v2")
    refined_v2.attrs["pose_schema"] = {"name": "traditional_v2"}

    _patch_scan(monkeypatch, {zarr_path: root})

    rc = mod.main([str(zarr_path), "--zarr-use", "training"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "already uses traditional_v2" in out
    assert "skipped_existing=1" in out
