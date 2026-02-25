from __future__ import annotations

import json
from pathlib import Path

from fisheye.utils import prune_legacy_eye_mask_profile_runs as mod


def _write_group_attrs(path: Path, attrs: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": attrs,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _make_profile_run(
    zarr_path: Path,
    run_name: str,
    *,
    stage_group: str | None,
    review_state: str | None = None,
    review_intended_use: str | None = None,
    eye_mask_method: str = "refine_eye_masks",
    eye_mask_run: str = "refine_eye_masks_001",
    created_at_utc: str = "2026-02-25T00:00:00+00:00",
) -> None:
    source: dict[str, object] = {
        "eye_mask_method": eye_mask_method,
        "eye_mask_run": eye_mask_run,
    }
    if stage_group is not None:
        source["stage_group"] = stage_group
    if review_state is not None:
        source["review_state"] = review_state
    if review_intended_use is not None:
        source["review_intended_use"] = review_intended_use
    summary = {
        "created_at_utc": created_at_utc,
        "source": source,
        "quality": {
            "rows_total": 100,
            "rows_usable": 95,
            "usable_rate": 0.95,
        },
    }
    _write_group_attrs(
        zarr_path / "analysis" / "eye_mask_profile_runs" / run_name / "zarr.json",
        {"profile_summary": summary},
    )


def _make_zarr(tmp_path: Path, *, latest: str = "new_profile") -> Path:
    zarr_path = tmp_path / "recording_training.zarr"
    _write_group_attrs(zarr_path / "zarr.json", {"zarr_use": "training"})
    _write_group_attrs(zarr_path / "analysis" / "zarr.json", {})
    _write_group_attrs(zarr_path / "analysis" / "eye_mask_profile_runs" / "zarr.json", {"latest": latest})
    return zarr_path


def test_plan_marks_legacy_candidate_with_replacement(tmp_path: Path) -> None:
    zarr_path = _make_zarr(tmp_path)
    _make_profile_run(zarr_path, "legacy_profile", stage_group=None)
    _make_profile_run(
        zarr_path,
        "new_profile",
        stage_group="refined_eye_masks_runs",
        review_state="approved",
        review_intended_use="training",
    )

    rows = mod._plan_zarr(zarr_path, zarr_use_filter="any", allow_unpaired=False)
    legacy_rows = [row for row in rows if row.profile_run == "legacy_profile"]
    assert len(legacy_rows) == 1
    row = legacy_rows[0]
    assert row.status == "legacy_candidate"
    assert row.action == "delete"
    assert row.replacement_run == "new_profile"


def test_plan_skips_latest_legacy_run_for_safety(tmp_path: Path) -> None:
    zarr_path = _make_zarr(tmp_path, latest="legacy_profile")
    _make_profile_run(zarr_path, "legacy_profile", stage_group=None)
    _make_profile_run(
        zarr_path,
        "new_profile",
        stage_group="refined_eye_masks_runs",
        review_state="approved",
        review_intended_use="training",
    )

    rows = mod._plan_zarr(zarr_path, zarr_use_filter="any", allow_unpaired=False)
    legacy_rows = [row for row in rows if row.profile_run == "legacy_profile"]
    assert len(legacy_rows) == 1
    row = legacy_rows[0]
    assert row.status == "legacy_latest"
    assert row.action == "skip"


def test_plan_skips_unpaired_legacy_by_default(tmp_path: Path) -> None:
    zarr_path = _make_zarr(tmp_path)
    _make_profile_run(zarr_path, "legacy_profile", stage_group=None)

    rows = mod._plan_zarr(zarr_path, zarr_use_filter="any", allow_unpaired=False)
    legacy_rows = [row for row in rows if row.profile_run == "legacy_profile"]
    assert len(legacy_rows) == 1
    row = legacy_rows[0]
    assert row.status == "legacy_no_replacement"
    assert row.action == "skip"


def test_plan_marks_missing_review_run_as_legacy_candidate(tmp_path: Path) -> None:
    zarr_path = _make_zarr(tmp_path)
    _make_profile_run(
        zarr_path,
        "legacy_missing_review",
        stage_group="refined_eye_masks_runs",
    )
    _make_profile_run(
        zarr_path,
        "new_profile",
        stage_group="refined_eye_masks_runs",
        review_state="approved",
        review_intended_use="training",
    )

    rows = mod._plan_zarr(zarr_path, zarr_use_filter="any", allow_unpaired=False)
    legacy_rows = [row for row in rows if row.profile_run == "legacy_missing_review"]
    assert len(legacy_rows) == 1
    row = legacy_rows[0]
    assert row.status == "legacy_candidate"
    assert row.action == "delete"
    assert row.replacement_run == "new_profile"


def test_apply_deletes_legacy_candidate(tmp_path: Path) -> None:
    zarr_path = _make_zarr(tmp_path)
    _make_profile_run(zarr_path, "legacy_profile", stage_group=None)
    _make_profile_run(
        zarr_path,
        "new_profile",
        stage_group="refined_eye_masks_runs",
        review_state="approved",
        review_intended_use="training",
    )

    rc = mod.main([str(zarr_path), "--apply"])
    assert rc == 0
    assert not (zarr_path / "analysis" / "eye_mask_profile_runs" / "legacy_profile").exists()
    assert (zarr_path / "analysis" / "eye_mask_profile_runs" / "new_profile").exists()
