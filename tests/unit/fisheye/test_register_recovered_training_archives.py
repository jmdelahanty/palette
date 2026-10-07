"""Recovered training archives are registered as tracked, never selectable."""

from __future__ import annotations

from pathlib import Path

import zarr

from fisheye.registry.db import Registry
from fisheye.utils import register_recovered_training_archives as tool
from fisheye.utils.report_acquisition_crop_video_roi_readiness import _is_training_row

REC = "2026-01-28T19-22-28Z_arena_1"


def _setup(tmp_path: Path, *, eligible: bool = False, originals: int = 1) -> tuple[Path, Path]:
    registry_path = tmp_path / "registry.sqlite"
    registry = Registry(registry_path)
    try:
        for i in range(originals):
            registry.upsert_dataset(
                f"{REC}:zorig{i}", session_uuid=REC, zarr_path=Path(f"/nvme1/recordings/{REC}_{i}_training.zarr"),
                recording_id=REC, artifact_kind="source_recording", zarr_use="training",
            )
        registry.conn.execute("UPDATE datasets SET status = 'missing'")
        registry.conn.commit()
    finally:
        registry.close()
    root = tmp_path / "recordings"
    archive = root / f"{REC}_DefaultScreen" / "zarr" / f"{REC}_DefaultScreen_recovered_training.zarr"
    group = zarr.open_group(str(archive), mode="w")
    group.attrs.update({"schema_id": tool.RECOVERY_SCHEMAS[1], "stage_selector_eligible": eligible,
                        "recording_id": REC, "zarr_purpose": "training", "recovery_mode": "source_only"})
    return registry_path, root


def test_archive_is_registered_as_recovered_with_lineage_to_its_missing_original(tmp_path):
    registry_path, root = _setup(tmp_path)
    planned, skipped = tool.plan(registry_path, root)
    assert len(planned) == 1 and skipped == []
    tool.apply_plan(registry_path, planned)

    registry = Registry(registry_path)
    try:
        row = registry.conn.execute(
            "SELECT zarr_use, artifact_kind, recording_id, status FROM datasets WHERE dataset_id = ?",
            (planned[0].dataset_id,)).fetchone()
        edge = registry.conn.execute(
            "SELECT parent_dataset_id, relationship_type FROM dataset_lineage WHERE child_dataset_id = ?",
            (planned[0].dataset_id,)).fetchone()
        original = registry.conn.execute(
            "SELECT status FROM datasets WHERE dataset_id = ?", (f"{REC}:zorig0",)).fetchone()
    finally:
        registry.close()
    assert tuple(row) == ("recovered_training", "recovered_training_derivative", REC, "active")
    assert tuple(edge) == (f"{REC}:zorig0", "recovered_from")
    assert original["status"] == "missing"
    assert tool.plan(registry_path, root)[1][0]["reason"] == "already registered"


def test_selector_eligible_or_ambiguous_archives_are_skipped(tmp_path):
    registry_path, root = _setup(tmp_path / "a", eligible=True)
    assert "selector-ineligible" in tool.plan(registry_path, root)[1][0]["reason"]
    registry_path, root = _setup(tmp_path / "b", originals=2)
    assert "2 missing original" in tool.plan(registry_path, root)[1][0]["reason"]


def test_readiness_report_does_not_count_recovered_archives_as_training():
    row = {"zarr_use": "recovered_training", "zarr_path": "/x/rec_recovered_training.zarr"}
    assert _is_training_row(row) is False
    assert _is_training_row({"zarr_use": "training", "zarr_path": "/x/rec_training.zarr"}) is True
