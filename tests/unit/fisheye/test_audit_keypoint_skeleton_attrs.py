from __future__ import annotations

import json
from pathlib import Path

from fisheye.utils import audit_keypoint_skeleton_attrs as mod
from fisheye.utils.audit_keypoint_skeleton_attrs import audit_keypoint_skeleton_attrs, main


class _FakeArray:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class _FakeGroup(dict):
    def __init__(self, *, attrs: dict | None = None) -> None:
        super().__init__()
        self.attrs = attrs or {}

    def get(self, key: str, default=None):  # noqa: A003
        return super().get(key, default)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_audit_keypoint_skeleton_attrs_reports_missing_explicit_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    zarr_path = tmp_path / "sample_analysis.zarr"
    root = _FakeGroup(attrs={"zarr_use": "analysis", "zarr_purpose": "analysis"})

    kp_ok = _FakeGroup(
        attrs={
            "skeleton_id": "pose_schema:traditional_v1",
            "kpt_shape": [3, 2],
            "pose_schema": {
                "name": "traditional_v1",
                "skeleton_id": "pose_schema:traditional_v1",
                "kpt_shape": [3, 2],
            },
        }
    )
    kp_ok["keypoints_roi"] = _FakeArray((2, 3, 2))

    refined_missing = _FakeGroup(attrs={"pose_schema": {"name": "traditional_v2"}})
    refined_missing["keypoints_roi"] = _FakeArray((2, 5, 2))

    monkeypatch.setattr(mod, "_iter_zarr", lambda roots, recursive: iter([zarr_path]))
    monkeypatch.setattr(mod, "open_zarr_root", lambda path, mode="r": root)
    monkeypatch.setattr(
        mod,
        "_iter_run_groups",
        lambda _root, all_runs, zarr_path=None, open_mode=None: iter(
            [
                ("keypoints_runs", "kp_ok", kp_ok),
                ("refined_keypoints_runs", "refined_missing", refined_missing),
            ]
        ),
    )

    rows = audit_keypoint_skeleton_attrs(
        [tmp_path],
        recursive=True,
        zarr_use="analysis",
        all_runs=True,
    )

    assert len(rows) == 2
    ok_row = next(row for row in rows if row["run_path"] == "keypoints_runs/kp_ok")
    missing_row = next(row for row in rows if row["run_path"] == "refined_keypoints_runs/refined_missing")

    assert ok_row["status"] == "ok"
    assert ok_row["missing_attrs"] == []
    assert missing_row["status"] == "missing_explicit_attrs"
    assert missing_row["missing_attrs"] == ["skeleton_id", "kpt_shape"]
    assert missing_row["resolved_skeleton_id"] == "pose_schema:traditional_v2"
    assert missing_row["resolved_kpt_shape"] == [5, 2]


def test_audit_keypoint_skeleton_attrs_main_strict_returns_nonzero(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        mod,
        "audit_keypoint_skeleton_attrs",
        lambda *args, **kwargs: [
            {
                "zarr_path": str(tmp_path / "sample_analysis.zarr"),
                "run_path": "refined_keypoints_runs/refined_missing",
                "status": "missing_explicit_attrs",
                "missing_attrs": ["skeleton_id", "kpt_shape"],
                "pose_schema_name": "traditional_v2",
                "resolved_skeleton_id": "pose_schema:traditional_v2",
                "resolved_kpt_shape": [5, 2],
                "zarr_use": "analysis",
            }
        ],
    )

    rc = main([str(tmp_path), "--recursive", "--all-runs", "--strict", "--no-log"])
    assert rc == 2


def test_audit_keypoint_skeleton_attrs_main_writes_jsonl_log(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    row = {
        "zarr_path": str(tmp_path / "sample_analysis.zarr"),
        "run_path": "keypoints_runs/keypoints_001",
        "status": "ok",
        "missing_attrs": [],
        "pose_schema_name": "traditional_v1",
        "resolved_skeleton_id": "pose_schema:traditional_v1",
        "resolved_kpt_shape": [3, 2],
        "zarr_use": "analysis",
    }
    monkeypatch.setattr(mod, "audit_keypoint_skeleton_attrs", lambda *args, **kwargs: [row])

    log_dir = tmp_path / "logs"
    rc = main([str(tmp_path), "--recursive", "--all-runs", "--log-dir", str(log_dir)])
    assert rc == 0

    out = capsys.readouterr().out
    assert "Log file:" in out
    assert "Keypoint skeleton attr audit: scope=all rows=1 ok=1 missing=0" in out

    log_files = sorted(log_dir.glob("audit_keypoint_skeleton_attrs_*.jsonl"))
    assert len(log_files) == 1

    rows = _read_jsonl(log_files[0])
    events = [str(item["event"]) for item in rows]
    assert events[0] == "run_start"
    assert "run_group_checked" in events
    assert events[-1] == "run_end"

    checked_row = next(item for item in rows if item["event"] == "run_group_checked")
    assert checked_row["run_path"] == "keypoints_runs/keypoints_001"
    assert checked_row["status"] == "ok"
    assert checked_row["zarr_path"] == str(tmp_path / "sample_analysis.zarr")

    end_row = rows[-1]
    assert end_row["status"] == "ok"
    assert end_row["mode"] == "text"
    assert end_row["rows"] == 1
    assert end_row["ok"] == 1
    assert end_row["missing"] == 0
