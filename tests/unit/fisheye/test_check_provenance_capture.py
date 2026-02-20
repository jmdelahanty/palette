from __future__ import annotations

from pathlib import Path
from typing import Any

from fisheye.diagnostics import check_provenance_capture as mod
from fisheye.shared.stage_provenance import (
    STAGE_PROVENANCE_CONTRACT_NAME,
    build_stage_provenance,
)


class _FakeGroup(dict):
    def __init__(self, *args: Any, attrs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.attrs = attrs or {}

    def group_keys(self) -> list[str]:
        return [key for key, value in self.items() if isinstance(value, _FakeGroup)]

    def get(self, key: str, default: Any = None) -> Any:
        return super().get(key, default)


def _build_root_for_stage(stage_group: str, run_name: str, run_attrs: dict[str, Any]) -> _FakeGroup:
    run = _FakeGroup(attrs=run_attrs)
    parent = _FakeGroup({run_name: run}, attrs={"latest": run_name})
    return _FakeGroup({stage_group: parent})


def test_stages_include_refined_eye_masks() -> None:
    labels = {stage["label"] for stage in mod.STAGES}
    assert "refined_eye_masks" in labels


def test_check_zarr_strict_contract_requires_refinement_contract_block(monkeypatch) -> None:
    root = _build_root_for_stage(
        "refined_eye_masks_runs",
        "refined_eye_masks_001",
        {
            "provenance": {
                "stage": "refine_eye_masks",
                "created_at_utc": "2026-02-20T00:00:00+00:00",
                "parameters": {"threshold": 0.5},
                "inputs": {"eye_masks_run": "eye_masks_001"},
                "git": {"commit": "a" * 40, "branch": "main"},
                "environment": {},
            }
        },
    )
    monkeypatch.setattr(mod.zarr, "open", lambda *_args, **_kwargs: root)

    checks, _, _ = mod._check_zarr(
        Path("/fake/recording.zarr"),
        all_runs=False,
        require_provenance=True,
        check_consistency=False,
        check_subject_metadata=False,
        strict_contract=True,
    )

    refined_checks = checks["refined_eye_masks"]
    assert len(refined_checks) == 1
    check = refined_checks[0]
    assert check.status == "missing"
    assert "contract" in check.missing_required


def test_check_zarr_strict_contract_accepts_canonical_refinement_contract(monkeypatch) -> None:
    payload = build_stage_provenance(
        stage="refine_eye_masks",
        created_at_utc="2026-02-20T00:00:00+00:00",
        parameters={"threshold": 0.5},
        inputs={"eye_masks_run": "eye_masks_001"},
        git={"commit": "b" * 40, "branch": "main"},
        environment={},
    )
    root = _build_root_for_stage(
        "refined_eye_masks_runs",
        "refined_eye_masks_001",
        {"provenance": payload},
    )
    monkeypatch.setattr(mod.zarr, "open", lambda *_args, **_kwargs: root)

    checks, _, _ = mod._check_zarr(
        Path("/fake/recording.zarr"),
        all_runs=False,
        require_provenance=True,
        check_consistency=False,
        check_subject_metadata=False,
        strict_contract=True,
    )

    check = checks["refined_eye_masks"][0]
    assert check.status == "ok"
    assert check.has_provenance is True
    assert check.missing_required == []
    assert STAGE_PROVENANCE_CONTRACT_NAME == payload["contract"]["name"]


def test_main_strict_returns_nonzero_on_refinement_contract_failure(monkeypatch, capsys) -> None:
    root = _build_root_for_stage(
        "refined_keypoints_runs",
        "refined_keypoints_001",
        {
            "provenance": {
                "stage": "refine_keypoints",
                "created_at_utc": "2026-02-20T00:00:00+00:00",
                "parameters": {"confidence_threshold": 0.2},
                "inputs": {"keypoints_run": "keypoints_001"},
            }
        },
    )
    fake_path = Path("/fake/recording.zarr")
    monkeypatch.setattr(mod.zarr, "open", lambda *_args, **_kwargs: root)
    monkeypatch.setattr(mod, "_iter_zarr", lambda *_args, **_kwargs: iter([fake_path]))

    rc = mod.main([str(fake_path), "--strict", "--no-check-consistency"])
    assert rc == 1
    capsys.readouterr()

    rc_non_strict = mod.main([str(fake_path), "--no-check-consistency"])
    assert rc_non_strict == 0


def test_main_strict_does_not_require_contract_for_non_refinement_stage(monkeypatch, capsys) -> None:
    root = _build_root_for_stage(
        "detect_runs",
        "detect_001",
        {
            "provenance": {
                "stage": "detect",
                "created_at_utc": "2026-02-20T00:00:00+00:00",
                "parameters": {"method": "yolo"},
                "inputs": {"source_video_path": "/tmp/video.mp4"},
                "git": {"commit": "c" * 40, "branch": "main"},
                "environment": {},
            }
        },
    )
    fake_path = Path("/fake/recording.zarr")
    monkeypatch.setattr(mod.zarr, "open", lambda *_args, **_kwargs: root)
    monkeypatch.setattr(mod, "_iter_zarr", lambda *_args, **_kwargs: iter([fake_path]))

    rc = mod.main([str(fake_path), "--strict", "--no-check-consistency"])
    assert rc == 0
    capsys.readouterr()
