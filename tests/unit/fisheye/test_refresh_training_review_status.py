from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

from fisheye.utils import refresh_training_review_status as mod


class _FakeRegistry:
    def __init__(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(
            """
            CREATE TABLE datasets (
                dataset_id TEXT,
                recording_id TEXT,
                zarr_path TEXT,
                zarr_use TEXT,
                status TEXT
            );
            """
        )


class _FakeGroup(dict[str, object]):
    def __init__(self, *, attrs: dict[str, object] | None = None) -> None:
        super().__init__()
        self.attrs = attrs or {}


def test_review_run_specs_use_deterministic_names_and_overrides() -> None:
    specs = mod._review_run_specs(
        "run_001",
        refined_subject_masks_run="manual_refined_subject_masks",
    )

    by_family = {spec.family: spec for spec in specs}
    assert by_family["keypoints"].run_name == "keypoints_training_review_run_001"
    assert by_family["refined_keypoints"].run_name == "refined_keypoints_training_review_run_001"
    assert by_family["subject_masks"].parent_group == "subject_mask_runs"
    assert by_family["subject_masks"].run_name == "subject_masks_training_review_run_001"
    assert by_family["refined_subject_masks"].run_name == "manual_refined_subject_masks"


def test_select_dataset_candidates_filters_training_scope_and_path(tmp_path: Path) -> None:
    registry = _FakeRegistry()
    inside = tmp_path / "recording_a" / "training.zarr"
    outside = tmp_path / "recording_b" / "training.zarr"
    registry.conn.executemany(
        "INSERT INTO datasets VALUES (?, ?, ?, ?, ?);",
        [
            ("ds_a", "rec_a", str(inside), "training", "ok"),
            ("ds_b", "rec_b", str(outside), "training", "ok"),
            ("ds_c", "rec_c", str(tmp_path / "analysis.zarr"), "analysis", "ok"),
            ("ds_missing", "rec_m", str(tmp_path / "missing.zarr"), "training", "missing"),
        ],
    )

    rows = mod._select_dataset_candidates(
        registry,  # type: ignore[arg-type]
        zarr_use="training",
        path_contains=("recording_a",),
        recording_ids=(),
        scope_paths=(tmp_path,),
    )

    assert [row.dataset_id for row in rows] == ["ds_a"]
    assert rows[0].recording_id == "rec_a"


def test_audit_and_stamp_review_runs_validates_before_marking(monkeypatch, tmp_path: Path) -> None:
    run_group = _FakeGroup(attrs={})
    parent = _FakeGroup(attrs={})
    parent["refined_subject_masks_training_review_run_001"] = run_group
    root = _FakeGroup()
    root["refined_subject_masks_runs"] = parent
    stamped: list[str] = []

    monkeypatch.setattr(mod, "_open_zarr_group", lambda *_args, **_kwargs: root)
    monkeypatch.setattr(mod, "validate_run", lambda *_args, **_kwargs: SimpleNamespace(valid=True))

    def _fake_mark_run_complete(  # type: ignore[no-untyped-def]
        group,
        *,
        parent_group,
        run_name,
        allow_missing_run_provenance=False,
        missing_run_provenance_reason=None,
    ):
        assert allow_missing_run_provenance is True
        assert missing_run_provenance_reason == "training review status refresh re-marks existing runs"
        stamped.append(run_name)
        group.attrs[mod.RUN_COMPLETION_STATUS_ATTR] = mod.RUN_STATUS_COMPLETE
        parent_group.attrs["latest_complete"] = run_name

    monkeypatch.setattr(mod, "mark_run_complete", _fake_mark_run_complete)

    summary = mod.audit_and_stamp_review_runs(
        [
            mod.DatasetCandidate(
                dataset_id="ds_a",
                recording_id="rec_a",
                zarr_path=tmp_path / "training.zarr",
                zarr_use="training",
            )
        ],
        [
            mod.ReviewRunSpec(
                family="refined_subject_masks",
                parent_group="refined_subject_masks_runs",
                stage_name="refined_subject_masks",
                run_name="refined_subject_masks_training_review_run_001",
            )
        ],
        apply=True,
        stamp_completion_markers=True,
    )

    counts = summary["families"]["refined_subject_masks"]  # type: ignore[index]
    assert counts["present"] == 1
    assert counts["valid"] == 1
    assert counts["stamped"] == 1
    assert stamped == ["refined_subject_masks_training_review_run_001"]
    assert parent.attrs["refined_subject_mask_review_status_latest"] == stamped[0]
