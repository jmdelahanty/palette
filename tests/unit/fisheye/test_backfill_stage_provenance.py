from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from fisheye.shared.stage_provenance import (
    STAGE_PROVENANCE_CONTRACT_NAME,
    STAGE_PROVENANCE_CONTRACT_VERSION,
)
from fisheye.utils import backfill_stage_provenance as mod


class _FakeGroup:
    def __init__(self, children: dict[str, "_FakeGroup"] | None = None) -> None:
        self._children: dict[str, _FakeGroup] = children or {}
        self.attrs: dict[str, Any] = {}

    def create_group(self, name: str) -> "_FakeGroup":
        child = _FakeGroup()
        self._children[name] = child
        return child

    def get(self, name: str) -> "_FakeGroup" | None:
        return self._children.get(name)

    def group_keys(self):  # pragma: no cover - exercised through prod helper
        return self._children.keys()

    def keys(self):  # pragma: no cover - exercised through prod helper
        return self._children.keys()

    def __contains__(self, key: str) -> bool:
        return key in self._children

    def __getitem__(self, key: str) -> "_FakeGroup":
        return self._children[key]


def _create_run(
    root: _FakeGroup,
    parent_name: str,
    run_name: str,
    *,
    attrs: dict[str, object],
) -> _FakeGroup:
    parent = root.get(parent_name)
    if parent is None:
        parent = root.create_group(parent_name)
    run = parent.create_group(run_name)
    for key, value in attrs.items():
        run.attrs[key] = value
    parent.attrs["latest"] = run_name
    return run


def _patch_scan(monkeypatch, mapping: dict[Path, _FakeGroup]) -> None:
    ordered_paths = list(mapping.keys())
    monkeypatch.setattr(mod, "_iter_zarr", lambda *_args, **_kwargs: iter(ordered_paths))

    def _open_group(path: str, mode: str = "r") -> _FakeGroup:  # noqa: ARG001
        return mapping[Path(path)]

    monkeypatch.setattr(mod.zarr, "open_group", _open_group)


def test_main_dry_run_reports_deterministic_counts_and_first_paths(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    zarr_path = tmp_path / "rec_analysis.zarr"
    root = _FakeGroup()
    root.attrs["zarr_purpose"] = "analysis"
    parent = root.create_group("detect_runs")
    for idx in range(1, 7):
        run = parent.create_group(f"detect_{idx:03d}")
        run.attrs["provenance"] = {
            "stage": "detect",
            "parameters": {"method": "yolo"},
            "inputs": {"source_video_path": "/tmp/video.mp4"},
            "git": {"branch": "main"},
        }
        run.attrs["git_commit_hash"] = "a" * 40
    parent.attrs["latest"] = "detect_006"
    _patch_scan(monkeypatch, {zarr_path: root})

    rc = mod.main([str(zarr_path), "--dry-run"])
    assert rc == 0

    out = capsys.readouterr().out
    assert "total_runs_scanned: 6" in out
    assert "missing_contract: 6" in out
    assert "missing_git_commit: 6" in out
    assert "would_modify_paths_first5:" in out
    for idx in range(1, 6):
        assert f"detect_runs/detect_{idx:03d}" in out
    assert "detect_runs/detect_006" not in out

    run = root["detect_runs"]["detect_001"]
    provenance = run.attrs["provenance"]
    assert "contract" not in provenance
    assert "commit" not in provenance["git"]


def test_main_apply_updates_only_contract_and_git_commit_from_commit_hash(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    zarr_path = tmp_path / "rec_analysis.zarr"
    root = _FakeGroup()
    root.attrs["zarr_purpose"] = "analysis"
    original_provenance = {
        "stage": "keypoints_detect",
        "parameters": {"method": "traditional_pose"},
        "inputs": {"source_crop_run": "crop_001"},
        "git": {"commit_hash": "b" * 40, "branch": "main"},
        "custom": {"keep": True},
    }
    _create_run(
        root,
        "keypoints_runs",
        "keypoints_001",
        attrs={"provenance": deepcopy(original_provenance)},
    )
    _patch_scan(monkeypatch, {zarr_path: root})

    rc = mod.main([str(zarr_path), "--apply"])
    assert rc == 0

    out = capsys.readouterr().out
    assert "total_runs_scanned: 1" in out
    assert "missing_contract: 1" in out
    assert "missing_git_commit: 1" in out
    assert "updated: 1" in out

    run = root["keypoints_runs"]["keypoints_001"]
    updated = run.attrs["provenance"]

    expected = deepcopy(original_provenance)
    expected["contract"] = {
        "name": STAGE_PROVENANCE_CONTRACT_NAME,
        "version": STAGE_PROVENANCE_CONTRACT_VERSION,
    }
    expected["git"]["commit"] = "b" * 40
    assert updated == expected
    assert sorted(run.attrs.keys()) == ["provenance"]


def test_main_apply_preserves_valid_contract_and_respects_zarr_use_filter(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    analysis_zarr = tmp_path / "rec_a" / "zarr" / "rec_a_analysis.zarr"
    analysis_root = _FakeGroup()
    analysis_root.attrs["zarr_purpose"] = "analysis"
    _create_run(
        analysis_root,
        "crop_runs",
        "crop_001",
        attrs={
            "provenance": {
                "stage": "crop",
                "parameters": {"pad": 4},
                "inputs": {"source_detect_run": "detect_001"},
                "contract": {
                    "name": STAGE_PROVENANCE_CONTRACT_NAME,
                    "version": 7,
                    "note": "keep",
                },
                "git": {"branch": "main"},
            },
            "git_commit": "c" * 40,
        },
    )

    training_zarr = tmp_path / "rec_b" / "zarr" / "rec_b_training.zarr"
    training_root = _FakeGroup()
    training_root.attrs["zarr_purpose"] = "training"
    _create_run(
        training_root,
        "crop_runs",
        "crop_001",
        attrs={
            "provenance": {
                "stage": "crop",
                "parameters": {"pad": 4},
                "inputs": {"source_detect_run": "detect_001"},
                "git": {"branch": "training"},
            },
            "git_commit": "d" * 40,
        },
    )
    _patch_scan(monkeypatch, {analysis_zarr: analysis_root, training_zarr: training_root})

    rc = mod.main([str(tmp_path), "--recursive", "--zarr-use", "analysis", "--apply"])
    assert rc == 0

    out = capsys.readouterr().out
    assert "total_runs_scanned: 1" in out
    assert "missing_contract: 0" in out
    assert "missing_git_commit: 1" in out
    assert "updated: 1" in out

    analysis_prov = analysis_root["crop_runs"]["crop_001"].attrs["provenance"]
    assert analysis_prov["contract"] == {
        "name": STAGE_PROVENANCE_CONTRACT_NAME,
        "version": 7,
        "note": "keep",
    }
    assert analysis_prov["git"]["commit"] == "c" * 40
    assert analysis_prov["git"]["branch"] == "main"

    training_prov = training_root["crop_runs"]["crop_001"].attrs["provenance"]
    assert "contract" not in training_prov
    assert "commit" not in training_prov["git"]
