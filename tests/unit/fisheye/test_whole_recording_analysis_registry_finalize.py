from __future__ import annotations

import json
from pathlib import Path

import pytest

from fisheye.cluster import whole_recording_analysis_registry_finalize as mod
from fisheye.cluster.whole_recording_analysis import PLAN_SCHEMA


class _Group(dict):
    def __init__(self, *, attrs: dict[str, object] | None = None) -> None:
        super().__init__()
        self.attrs = dict(attrs or {})


class _Connection:
    def execute(self, sql: str, _params=()):  # noqa: ANN001
        assert "integrity_check" in sql
        return self

    def fetchone(self):
        return ("ok",)


class _Registry:
    def __init__(self, _path: Path) -> None:
        self.conn = _Connection()

    def close(self) -> None:
        return None


def _write_plan(tmp_path: Path) -> Path:
    run_root = tmp_path / "combined"
    run_root.mkdir()
    keypoint_plan = tmp_path / "keypoints" / "plan.json"
    keypoint_plan.parent.mkdir()
    keypoint_plan.write_text("{}", encoding="utf-8")
    (run_root / "plan.json").write_text(
        json.dumps(
            {
                "schema": PLAN_SCHEMA,
                "keypoint_plan_path": str(keypoint_plan),
                "targets": [
                    {
                        "target_id": "target_a",
                        "analysis_zarr": str(tmp_path / "analysis.zarr"),
                        "refined_keypoint_run": "refined_keypoints_exact",
                        "subject_masks": {
                            "subject_mask_run": "subject_exact",
                            "refined_subject_mask_run": "refined_subject_exact",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return run_root


def _root(*, assignment_run: str = "refined_keypoints_exact") -> _Group:
    root = _Group()
    subject_parent = _Group()
    subject_parent["subject_exact"] = _Group()
    refined_parent = _Group()
    refined_parent["refined_subject_exact"] = _Group(
        attrs={
            "assignment_keypoint_group": "refined_keypoints_runs",
            "assignment_keypoints_run": assignment_run,
        }
    )
    root["subject_mask_runs"] = subject_parent
    root["refined_subject_masks_runs"] = refined_parent
    return root


def test_combined_registry_finalizer_validates_exact_mask_lineage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root = _write_plan(tmp_path)
    keypoint_calls: list[bool] = []

    def validate_keypoints(*_args, **kwargs):
        keypoint_calls.append(bool(kwargs["apply"]))
        return {"status": "ok"}

    monkeypatch.setattr(mod, "finalize_keypoints", validate_keypoints)
    monkeypatch.setattr(mod, "open_zarr_group_direct", lambda *_args, **_kwargs: _root())
    monkeypatch.setattr(mod, "is_run_complete_in_parent", lambda *_args: True)
    monkeypatch.setattr(mod, "Registry", _Registry)

    report = mod.finalize_registry(
        run_root,
        registry_path=tmp_path / "registry.sqlite",
        apply=False,
    )

    assert report["status"] == "ok"
    assert keypoint_calls == [False]
    assert report["registry_integrity"] == "ok"
    assert report["subject_masks"][0]["assignment_keypoint_run"] == (
        "refined_keypoints_exact"
    )


def test_combined_registry_finalizer_rejects_wrong_assignment_lineage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root = _write_plan(tmp_path)
    monkeypatch.setattr(mod, "finalize_keypoints", lambda *_args, **_kwargs: {"status": "ok"})
    monkeypatch.setattr(
        mod,
        "open_zarr_group_direct",
        lambda *_args, **_kwargs: _root(assignment_run="wrong_run"),
    )
    monkeypatch.setattr(mod, "is_run_complete_in_parent", lambda *_args: True)

    with pytest.raises(RuntimeError, match="lineage mismatch"):
        mod.finalize_registry(
            run_root,
            registry_path=tmp_path / "registry.sqlite",
            apply=False,
        )
