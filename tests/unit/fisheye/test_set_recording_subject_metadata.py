from __future__ import annotations

import json
from pathlib import Path

import pytest
import zarr

from fisheye.shared.experiment_setup import resolve_experiment_setup
from fisheye.shared.subject_metadata import resolve_subject_metadata
from fisheye.shared.source_recording_identity import (
    SOURCE_RECORDING_IDENTITY_PROFILE,
    SOURCE_RECORDING_IDENTITY_PROFILE_ATTR,
)
from fisheye.utils import set_recording_subject_metadata as setter


def _recording(tmp_path: Path) -> Path:
    recording_dir = tmp_path / "Cam2010093"
    zarr_path = recording_dir / "zarr" / "Cam2010093_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w", zarr_format=3)
    root.attrs.update(
        {
            "session_uuid": recording_dir.name,
            "recording_id": recording_dir.name,
            "recording_name": recording_dir.name,
            "recording_type": "behavior",
            "experiment_context_status": "absent",
            "zarr_purpose": "analysis",
        }
    )
    recording_dir.mkdir(parents=True, exist_ok=True)
    (recording_dir / "recording_manifest.json").write_text(
        json.dumps({"recording_name": recording_dir.name}),
        encoding="utf-8",
    )
    return recording_dir


def test_apply_publishes_count_only_authorities_and_manifest_audit(tmp_path: Path) -> None:
    recording_dir = _recording(tmp_path)
    plan = setter.plan_recording(
        recording_dir,
        species="Danionella cerebrum",
        dpf=7,
        subject_count=5,
    )

    result = setter.apply_plan(
        plan,
        repair_id="test_repair",
        reason="known acquisition metadata",
        registry=None,
    )

    assert result["status"] == "applied"
    manifest = json.loads(
        (recording_dir / "recording_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["species"] == "Danionella cerebrum"
    assert manifest["dpf_at_acquisition"] == 7
    assert manifest["subject_count"] == 5
    assert manifest["metadata_repairs"][-1]["repair_id"] == "test_repair"

    root = zarr.open_group(
        str(recording_dir / "zarr" / "Cam2010093_analysis.zarr"),
        mode="r",
        use_consolidated=False,
    )
    subject = resolve_subject_metadata(root, allow_legacy=False)
    setup = resolve_experiment_setup(root, allow_legacy=False)
    assert subject.metadata["species"] == "Danionella cerebrum"
    assert subject.metadata["days_post_fertilization"] == 7
    assert subject.subject_ids == ()
    assert subject.subject_identity_kind == "none"
    assert setup.expected_subject_count == 5
    assert setup.assigned_subject_count is None
    assert setup.subject_assignment_status == "count_only"
    assert setup.source["kind"] == "recording_manifest_subject_metadata"
    assert root.attrs["subject_count"] == 5
    assert root.attrs["species"] == "Danionella cerebrum"


def test_apply_fences_current_profile_before_manifest_or_zarr_mutation(
    tmp_path: Path,
) -> None:
    recording_dir = _recording(tmp_path)
    plan = setter.plan_recording(
        recording_dir,
        species="Danionella cerebrum",
        dpf=7,
        subject_count=5,
    )
    zarr_path = recording_dir / "zarr" / "Cam2010093_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="r+", use_consolidated=False)
    root.attrs[SOURCE_RECORDING_IDENTITY_PROFILE_ATTR] = (
        SOURCE_RECORDING_IDENTITY_PROFILE
    )
    manifest_path = recording_dir / "recording_manifest.json"
    manifest_before = manifest_path.read_bytes()

    with pytest.raises(ValueError, match="current-profile"):
        setter.apply_plan(
            plan,
            repair_id="test_repair",
            reason="known acquisition metadata",
            registry=None,
        )

    assert manifest_path.read_bytes() == manifest_before
    reopened = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    assert "analysis/subject_metadata_runs" not in reopened
    assert "analysis/experiment_setup_runs" not in reopened


def test_plan_rejects_conflicting_manifest_metadata(tmp_path: Path) -> None:
    recording_dir = _recording(tmp_path)
    manifest_path = recording_dir / "recording_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["species"] = "Danio rerio"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    plan = setter.plan_recording(
        recording_dir,
        species="Danionella cerebrum",
        dpf=7,
        subject_count=5,
    )

    assert plan["status"] == "conflict"
    assert any("species" in conflict for conflict in plan["conflicts"])
