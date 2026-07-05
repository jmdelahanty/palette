from __future__ import annotations

import json
from pathlib import Path

from fisheye.utils import backfill_keypoint_skeleton_attrs as mod
from fisheye.utils.backfill_keypoint_skeleton_attrs import _backfill_run_group, main


class _FakeArray:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class _FakeGroup(dict):
    def __init__(self, *args, attrs: dict | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attrs = attrs or {}

    def get(self, key: str, default=None):  # noqa: A003
        return super().get(key, default)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_backfill_run_group_populates_explicit_identity_and_pose_schema() -> None:
    run = _FakeGroup(
        attrs={
            "pose_schema": {
                "name": "traditional_v1",
                "metadata": {},
            }
        }
    )
    run["keypoints_roi"] = _FakeArray((2, 3, 2))

    result = _backfill_run_group(run, apply=True)

    assert result.status == "ok"
    assert run.attrs["skeleton_id"] == "pose_schema:traditional_v1"
    assert run.attrs["kpt_shape"] == [3, 2]
    assert run.attrs["pose_schema"]["skeleton_id"] == "pose_schema:traditional_v1"
    assert run.attrs["pose_schema"]["kpt_shape"] == [3, 2]


def test_backfill_run_group_prefers_explicit_identity_when_normalizing_pose_schema() -> None:
    run = _FakeGroup(
        attrs={
            "skeleton_id": "explicit_skeleton",
            "kpt_shape": [5, 2],
            "pose_schema": {
                "name": "traditional_v2",
                "skeleton_id": "pose_schema:traditional_v2",
                "metadata": {},
            },
        }
    )
    run["keypoints_roi"] = _FakeArray((2, 5, 2))

    result = _backfill_run_group(run, apply=True)

    assert result.status == "ok"
    assert run.attrs["pose_schema"]["skeleton_id"] == "explicit_skeleton"
    assert run.attrs["pose_schema"]["kpt_shape"] == [5, 2]


def test_backfill_run_group_skips_when_identity_attrs_already_present() -> None:
    run = _FakeGroup(
        attrs={
            "skeleton_id": "pose_schema:traditional_v1",
            "kpt_shape": [3, 2],
            "pose_schema": {
                "name": "traditional_v1",
                "skeleton_id": "pose_schema:traditional_v1",
                "kpt_shape": [3, 2],
                "metadata": {},
            },
        }
    )
    run["keypoints_roi"] = _FakeArray((2, 3, 2))

    result = _backfill_run_group(run, apply=True)

    assert result.status == "skipped_existing"


def test_backfill_run_group_reports_missing_pose_schema() -> None:
    run = _FakeGroup(attrs={})
    run["keypoints_roi"] = _FakeArray((2, 3, 2))

    result = _backfill_run_group(run, apply=True)

    assert result.status == "no_pose_schema"


def test_iter_run_groups_includes_direct_fs_run_names(monkeypatch) -> None:
    root = _FakeGroup(
        attrs={},
        keypoints_runs=_FakeGroup(
            attrs={"latest": "keypoints_001"},
            keypoints_001=_FakeGroup(attrs={"name": "embedded"}),
        ),
    )
    direct_groups = {
        "keypoints_001": _FakeGroup(attrs={"name": "direct-001"}),
        "keypoints_002": _FakeGroup(attrs={"name": "direct-002"}),
    }
    zarr_path = Path("/tmp/fake_training.zarr")
    seen_modes: list[str] = []

    monkeypatch.setattr(mod, "direct_zarr_group_names", lambda path: ["keypoints_001", "keypoints_002"])
    monkeypatch.setattr(
        mod,
        "open_zarr_group_direct",
        lambda path, mode: seen_modes.append(mode) or direct_groups[Path(path).name],
    )

    groups = list(mod._iter_run_groups(root, all_runs=True, zarr_path=zarr_path, open_mode="a"))

    assert len(groups) == 2
    assert groups[0][1] is direct_groups["keypoints_001"]
    assert groups[1][1] is direct_groups["keypoints_002"]
    assert seen_modes == ["a", "a"]


def test_main_writes_jsonl_log(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        mod,
        "_iter_zarr",
        lambda roots, recursive: iter([tmp_path / "sample_training.zarr"]),
    )
    monkeypatch.setattr(mod, "open_zarr_root", lambda path, mode="r": _FakeGroup(attrs={"zarr_purpose": "training"}))
    run = _FakeGroup(attrs={"pose_schema": {"name": "traditional_v1", "metadata": {}}})
    run["keypoints_roi"] = _FakeArray((2, 3, 2))
    monkeypatch.setattr(
        mod,
        "_iter_run_groups",
        lambda root, all_runs, zarr_path=None, open_mode=None: iter([("keypoints_runs/keypoints_001", run)]),
    )

    log_dir = tmp_path / "logs"
    rc = main([str(tmp_path), "--zarr-use", "any", "--log-dir", str(log_dir)])
    assert rc == 0

    out = capsys.readouterr().out
    assert "Log file:" in out
    assert "Dry run: ok=1" in out

    log_files = sorted(log_dir.glob("backfill_keypoint_skeleton_attrs_*.jsonl"))
    assert len(log_files) == 1

    rows = _read_jsonl(log_files[0])
    events = [str(item["event"]) for item in rows]
    assert events[0] == "run_start"
    assert "run_group_checked" in events
    assert events[-1] == "run_end"

    checked_row = next(item for item in rows if item["event"] == "run_group_checked")
    assert checked_row["run_path"] == "keypoints_runs/keypoints_001"
    assert checked_row["status"] == "ok"
    assert checked_row["changed"] is True
    assert checked_row["resolved_skeleton_id"] == "pose_schema:traditional_v1"
    assert checked_row["resolved_kpt_shape"] == [3, 2]

    end_row = rows[-1]
    assert end_row["mode"] == "dry-run"
    assert end_row["runs_considered"] == 1
    assert end_row["ok"] == 1
