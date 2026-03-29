from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

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


def _minimal_provenance(stage_name: str) -> dict[str, Any]:
    return {
        "stage": stage_name,
        "created_at_utc": "2026-02-20T00:00:00+00:00",
        "parameters": {"alpha": 1},
        "inputs": {"source_crop_run": "crop_001"},
        "git": {"commit": "a" * 40, "branch": "main"},
        "environment": {},
    }


def test_stages_include_subject_and_refined_mask_stages() -> None:
    labels = {stage["label"] for stage in mod.STAGES}
    assert "subject_masks" in labels
    assert "refined_subject_masks" in labels
    assert "refined_eye_masks" in labels


@pytest.mark.parametrize(
    ("stage_group", "stage_label", "run_name", "stage_name"),
    [
        ("detect_runs", "detect", "detect_001", "detect"),
        ("refined_detect_runs", "refined_detect", "refined_detect_001", "refine_detect"),
        ("crop_runs", "crop", "crop_001", "crop"),
        ("keypoints_runs", "keypoints", "keypoints_001", "keypoints_detect"),
        ("refined_keypoints_runs", "refined_keypoints", "refined_keypoints_001", "refine_keypoints"),
        ("eye_masks_runs", "eye_masks", "eye_masks_001", "eye_masks"),
        ("refined_eye_masks_runs", "refined_eye_masks", "refined_eye_masks_001", "refine_eye_masks"),
        ("subject_mask_runs", "subject_masks", "subject_masks_001", "subject_masks"),
        ("refined_subject_masks_runs", "refined_subject_masks", "refined_subject_masks_001", "refine_subject_masks"),
        ("arena_assignment_runs", "arena_assignment", "arena_assignment_001", "arena_assignment"),
    ],
)
def test_check_zarr_strict_contract_requires_contract_for_migrated_offline_stages(
    monkeypatch,
    stage_group: str,
    stage_label: str,
    run_name: str,
    stage_name: str,
) -> None:
    root = _build_root_for_stage(
        stage_group,
        run_name,
        {"provenance": _minimal_provenance(stage_name)},
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

    check = checks[stage_label][0]
    assert check.status == "missing"
    assert "contract" in check.missing_required


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


@pytest.mark.parametrize(
    ("contract_payload", "expected_field"),
    [
        ({"version": 1}, "contract.name"),
        ({"name": STAGE_PROVENANCE_CONTRACT_NAME}, "contract.version"),
        ({"name": "wrong_contract", "version": 1}, "contract.name"),
        ({"name": STAGE_PROVENANCE_CONTRACT_NAME, "version": 0}, "contract.version"),
    ],
)
def test_check_zarr_strict_contract_reports_missing_contract_fields(
    monkeypatch,
    contract_payload: dict[str, Any],
    expected_field: str,
) -> None:
    provenance = _minimal_provenance("detect")
    provenance["contract"] = contract_payload
    root = _build_root_for_stage("detect_runs", "detect_001", {"provenance": provenance})
    monkeypatch.setattr(mod.zarr, "open", lambda *_args, **_kwargs: root)

    checks, _, _ = mod._check_zarr(
        Path("/fake/recording.zarr"),
        all_runs=False,
        require_provenance=True,
        check_consistency=False,
        check_subject_metadata=False,
        strict_contract=True,
    )

    check = checks["detect"][0]
    assert check.status == "missing"
    assert expected_field in check.missing_required


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


def test_main_strict_returns_nonzero_on_detect_contract_failure(monkeypatch, capsys) -> None:
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
    assert rc == 1
    capsys.readouterr()


def test_main_zarr_use_filter_runs_checks_only_for_matching_archives(monkeypatch, capsys) -> None:
    analysis_path = Path("/fake/recording_analysis.zarr")
    training_path = Path("/fake/recording_training.zarr")
    analysis_root = _FakeGroup(attrs={"zarr_use": "analysis"})
    training_root = _FakeGroup(attrs={"zarr_use": "training"})
    root_lookup = {
        analysis_path: analysis_root,
        training_path: training_root,
    }

    monkeypatch.setattr(mod, "_iter_zarr", lambda *_args, **_kwargs: iter([analysis_path, training_path]))
    monkeypatch.setattr(mod.zarr, "open", lambda path, mode="r": root_lookup[Path(path)])  # noqa: ARG005

    seen: list[Path] = []

    def _fake_check_zarr(zarr_path: Path, **kwargs: Any):
        seen.append(zarr_path)
        assert kwargs["root"] is root_lookup[zarr_path]
        checks = {
            stage["label"]: [mod.ProvenanceCheck(stage["label"], "run", "ok", True, [], [])]
            for stage in mod.STAGES
        }
        return checks, None, None

    monkeypatch.setattr(mod, "_check_zarr", _fake_check_zarr)

    rc = mod.main(
        [
            str(analysis_path),
            str(training_path),
            "--zarr-use",
            "analysis",
            "--no-check-consistency",
        ]
    )
    assert rc == 0
    assert seen == [analysis_path]
    capsys.readouterr()


def test_main_strict_keeps_legacy_stage_compatible_until_backfill(monkeypatch, capsys) -> None:
    root = _build_root_for_stage(
        "detect_runs",
        "detect_legacy_001",
        {
            "provenance": {
                "stage": "detect_legacy",
                "created_at_utc": "2026-02-20T00:00:00+00:00",
                "parameters": {"method": "legacy"},
                "inputs": {"source_video_path": "/tmp/video.mp4"},
                "git": {"commit": "d" * 40, "branch": "main"},
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


def test_check_zarr_accepts_canonical_contract_payload_for_eye_masks_stage(monkeypatch) -> None:
    payload = build_stage_provenance(
        stage="eye_masks",
        created_at_utc="2026-02-20T00:00:00+00:00",
        parameters={"method": "traditional_eye_segmentation"},
        inputs={"source_crop_run": "crop_001"},
        git={"commit": "d" * 40, "branch": "main"},
        environment={},
    )
    root = _build_root_for_stage(
        "eye_masks_runs",
        "eye_masks_001",
        {
            "method": "traditional_eye_segmentation",
            "provenance": payload,
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

    check = checks["eye_masks"][0]
    assert check.status == "ok"
    assert check.has_provenance is True


@pytest.mark.parametrize(
    ("stage_group", "stage_label", "run_name", "stage_name"),
    [
        ("subject_mask_runs", "subject_masks", "subject_masks_001", "subject_masks"),
        ("refined_subject_masks_runs", "refined_subject_masks", "refined_subject_masks_001", "refine_subject_masks"),
    ],
)
def test_check_zarr_accepts_canonical_contract_payload_for_subject_mask_stages(
    monkeypatch,
    stage_group: str,
    stage_label: str,
    run_name: str,
    stage_name: str,
) -> None:
    payload = build_stage_provenance(
        stage=stage_name,
        created_at_utc="2026-02-20T00:00:00+00:00",
        parameters={"method": "subject_masks_test"},
        inputs={"source_crop_run": "crop_001"},
        git={"commit": "d" * 40, "branch": "main"},
        environment={},
    )
    root = _build_root_for_stage(
        stage_group,
        run_name,
        {
            "method": "subject_masks_test",
            "provenance": payload,
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

    check = checks[stage_label][0]
    assert check.status == "ok"
    assert check.has_provenance is True


def test_check_zarr_accepts_canonical_contract_payload_for_keypoints_stage(monkeypatch) -> None:
    payload = build_stage_provenance(
        stage="keypoints_detect",
        created_at_utc="2026-02-20T00:00:00+00:00",
        parameters={"method": "traditional_pose"},
        inputs={"source_crop_run": "crop_001", "source_detect_run": "detect_001"},
        git={"commit": "e" * 40, "branch": "main"},
        environment={},
    )
    root = _build_root_for_stage(
        "keypoints_runs",
        "keypoints_001",
        {
            "method": "traditional_pose",
            "provenance": payload,
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

    check = checks["keypoints"][0]
    assert check.status == "ok"
    assert check.has_provenance is True


def test_check_zarr_accepts_canonical_contract_payload_for_detect_stage(monkeypatch) -> None:
    payload = build_stage_provenance(
        stage="detect",
        created_at_utc="2026-02-20T00:00:00+00:00",
        parameters={"method": "yolo"},
        inputs={"source_video_path": "/tmp/video.mp4"},
        git={"commit": "f" * 40, "branch": "main"},
        environment={},
    )
    root = _build_root_for_stage(
        "detect_runs",
        "detect_001",
        {
            "detection_method": "yolo",
            "provenance": payload,
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

    check = checks["detect"][0]
    assert check.status == "ok"
    assert check.has_provenance is True
