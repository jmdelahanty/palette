from __future__ import annotations

from pathlib import Path

import zarr

from fisheye.shared.provenance_attrs import (
    CANONICAL_SOURCE_KEYPOINTS_RUN_ATTR,
    LEGACY_SOURCE_KEYPOINT_RUN_ATTR,
)
from fisheye.utils import backfill_eye_mask_lineage_attrs as mod


def _seed_archive(
    zarr_path: Path,
    *,
    zarr_use: str,
    eye_attrs: dict[str, object] | None = None,
    refined_attrs: dict[str, object] | None = None,
    keypoint_runs: dict[str, str] | None = None,
) -> None:
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs["zarr_purpose"] = zarr_use

    if keypoint_runs:
        for group_name, run_name in keypoint_runs.items():
            parent = root.require_group(group_name)
            parent.create_group(run_name)

    eye_parent = root.create_group("eye_masks_runs")
    eye = eye_parent.create_group("eye_masks_001")
    if eye_attrs:
        for key, value in eye_attrs.items():
            eye.attrs[key] = value
    eye_parent.attrs["latest"] = "eye_masks_001"

    refined_parent = root.create_group("refined_eye_masks_runs")
    refined = refined_parent.create_group("refined_eye_masks_001")
    if refined_attrs:
        for key, value in refined_attrs.items():
            refined.attrs[key] = value
    refined_parent.attrs["latest"] = "refined_eye_masks_001"


def test_backfill_eye_mask_lineage_dry_run_reports_without_writing(tmp_path: Path, capsys) -> None:
    zarr_path = tmp_path / "sample_analysis.zarr"
    _seed_archive(
        zarr_path,
        zarr_use="analysis",
        eye_attrs={LEGACY_SOURCE_KEYPOINT_RUN_ATTR: "kp_legacy_eye"},
        refined_attrs={LEGACY_SOURCE_KEYPOINT_RUN_ATTR: "kp_legacy_refined"},
    )

    rc = mod.main([str(zarr_path)])
    assert rc == 0

    root = zarr.open_group(str(zarr_path), mode="r")
    eye = root["eye_masks_runs"]["eye_masks_001"]
    refined = root["refined_eye_masks_runs"]["refined_eye_masks_001"]
    assert CANONICAL_SOURCE_KEYPOINTS_RUN_ATTR not in eye.attrs
    assert CANONICAL_SOURCE_KEYPOINTS_RUN_ATTR not in refined.attrs

    out = capsys.readouterr().out
    assert "Dry run: updated=0 would_update=2" in out


def test_backfill_eye_mask_lineage_apply_updates_both_archives_by_default(tmp_path: Path, capsys) -> None:
    analysis_zarr = tmp_path / "a_analysis.zarr"
    training_zarr = tmp_path / "b_training.zarr"

    _seed_archive(
        analysis_zarr,
        zarr_use="analysis",
        eye_attrs={LEGACY_SOURCE_KEYPOINT_RUN_ATTR: "kp_analysis_eye"},
        refined_attrs={LEGACY_SOURCE_KEYPOINT_RUN_ATTR: "kp_analysis_refined"},
        keypoint_runs={
            "keypoints_runs": "kp_analysis_eye",
            "refined_keypoints_runs": "kp_analysis_refined",
        },
    )
    _seed_archive(
        training_zarr,
        zarr_use="training",
        eye_attrs={LEGACY_SOURCE_KEYPOINT_RUN_ATTR: "kp_training_eye"},
        refined_attrs={LEGACY_SOURCE_KEYPOINT_RUN_ATTR: "kp_training_refined"},
        keypoint_runs={
            "keypoints_runs": "kp_training_eye",
            "refined_keypoints_runs": "kp_training_refined",
        },
    )

    rc = mod.main([str(tmp_path), "--recursive", "--apply"])
    assert rc == 0

    analysis_root = zarr.open_group(str(analysis_zarr), mode="r")
    training_root = zarr.open_group(str(training_zarr), mode="r")
    assert (
        analysis_root["eye_masks_runs"]["eye_masks_001"].attrs[CANONICAL_SOURCE_KEYPOINTS_RUN_ATTR]
        == "kp_analysis_eye"
    )
    assert analysis_root["eye_masks_runs"]["eye_masks_001"].attrs["source_keypoint_group"] == "keypoints_runs"
    assert (
        analysis_root["refined_eye_masks_runs"]["refined_eye_masks_001"].attrs[
            CANONICAL_SOURCE_KEYPOINTS_RUN_ATTR
        ]
        == "kp_analysis_refined"
    )
    assert (
        analysis_root["refined_eye_masks_runs"]["refined_eye_masks_001"].attrs["source_keypoint_group"]
        == "refined_keypoints_runs"
    )
    assert (
        training_root["eye_masks_runs"]["eye_masks_001"].attrs[CANONICAL_SOURCE_KEYPOINTS_RUN_ATTR]
        == "kp_training_eye"
    )
    assert (
        training_root["refined_eye_masks_runs"]["refined_eye_masks_001"].attrs[
            CANONICAL_SOURCE_KEYPOINTS_RUN_ATTR
        ]
        == "kp_training_refined"
    )

    out = capsys.readouterr().out
    assert "Applied: updated=4" in out


def test_backfill_eye_mask_lineage_analysis_only_skips_training(tmp_path: Path, capsys) -> None:
    analysis_zarr = tmp_path / "a_analysis.zarr"
    training_zarr = tmp_path / "b_training.zarr"

    _seed_archive(
        analysis_zarr,
        zarr_use="analysis",
        eye_attrs={LEGACY_SOURCE_KEYPOINT_RUN_ATTR: "kp_analysis"},
        refined_attrs={LEGACY_SOURCE_KEYPOINT_RUN_ATTR: "kp_analysis_refined"},
        keypoint_runs={
            "keypoints_runs": "kp_analysis",
            "refined_keypoints_runs": "kp_analysis_refined",
        },
    )
    _seed_archive(
        training_zarr,
        zarr_use="training",
        eye_attrs={LEGACY_SOURCE_KEYPOINT_RUN_ATTR: "kp_training"},
        refined_attrs={LEGACY_SOURCE_KEYPOINT_RUN_ATTR: "kp_training_refined"},
        keypoint_runs={
            "keypoints_runs": "kp_training",
            "refined_keypoints_runs": "kp_training_refined",
        },
    )

    rc = mod.main([str(tmp_path), "--recursive", "--analysis-only", "--apply"])
    assert rc == 0

    analysis_root = zarr.open_group(str(analysis_zarr), mode="r")
    training_root = zarr.open_group(str(training_zarr), mode="r")
    assert (
        analysis_root["eye_masks_runs"]["eye_masks_001"].attrs[CANONICAL_SOURCE_KEYPOINTS_RUN_ATTR]
        == "kp_analysis"
    )
    assert CANONICAL_SOURCE_KEYPOINTS_RUN_ATTR not in training_root["eye_masks_runs"]["eye_masks_001"].attrs

    out = capsys.readouterr().out
    assert "scope=analysis" in out
    assert "filtered_zarr_use=1" in out


def test_backfill_eye_mask_lineage_reports_conflicts_without_overwrite(tmp_path: Path, capsys) -> None:
    zarr_path = tmp_path / "conflict_training.zarr"
    _seed_archive(
        zarr_path,
        zarr_use="training",
        eye_attrs={
            CANONICAL_SOURCE_KEYPOINTS_RUN_ATTR: "kp_new",
            LEGACY_SOURCE_KEYPOINT_RUN_ATTR: "kp_old",
        },
        refined_attrs={
            CANONICAL_SOURCE_KEYPOINTS_RUN_ATTR: "kp_new_refined",
            LEGACY_SOURCE_KEYPOINT_RUN_ATTR: "kp_old_refined",
        },
    )

    rc = mod.main([str(zarr_path), "--apply"])
    assert rc == 0

    root = zarr.open_group(str(zarr_path), mode="r")
    eye = root["eye_masks_runs"]["eye_masks_001"]
    refined = root["refined_eye_masks_runs"]["refined_eye_masks_001"]
    assert eye.attrs[CANONICAL_SOURCE_KEYPOINTS_RUN_ATTR] == "kp_new"
    assert eye.attrs[LEGACY_SOURCE_KEYPOINT_RUN_ATTR] == "kp_old"
    assert refined.attrs[CANONICAL_SOURCE_KEYPOINTS_RUN_ATTR] == "kp_new_refined"
    assert refined.attrs[LEGACY_SOURCE_KEYPOINT_RUN_ATTR] == "kp_old_refined"

    out = capsys.readouterr().out
    assert "conflict=2" in out
    assert "Conflicts (first 5):" in out


def test_backfill_eye_mask_lineage_explicit_override_repairs_stale_source_run(tmp_path: Path, capsys) -> None:
    zarr_path = tmp_path / "repair_analysis.zarr"
    _seed_archive(
        zarr_path,
        zarr_use="analysis",
        eye_attrs={LEGACY_SOURCE_KEYPOINT_RUN_ATTR: "kp_missing"},
        refined_attrs={},
        keypoint_runs={"keypoints_runs": "kp_valid"},
    )

    rc = mod.main(
        [
            str(zarr_path),
            "--apply",
            "--run-path",
            "eye_masks_runs/eye_masks_001",
            "--keypoint-run",
            "kp_valid",
            "--keypoint-group",
            "keypoints_runs",
        ]
    )
    assert rc == 0

    root = zarr.open_group(str(zarr_path), mode="r")
    eye = root["eye_masks_runs"]["eye_masks_001"]
    assert eye.attrs[CANONICAL_SOURCE_KEYPOINTS_RUN_ATTR] == "kp_valid"
    assert eye.attrs[LEGACY_SOURCE_KEYPOINT_RUN_ATTR] == "kp_valid"
    assert eye.attrs["source_keypoint_group"] == "keypoints_runs"

    out = capsys.readouterr().out
    assert "updated=1" in out
