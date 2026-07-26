from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import zarr

from fisheye.analysis_workflows.materializers import arena_geometry_candidates as mod
from fisheye.shared.json_safety import strict_json_dumps
from fisheye.shared.recording_geometry import (
    BoundRegisteredDishMask,
    CircleGeometry,
    CitrusRegistrationStatus,
    MaterializedAssetStatus,
    RecordingGeometryError,
    RegisteredDishMask,
    RegisteredDishMaskKey,
)
from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.shared.zarr_io import open_zarr_root


def _bound_mask() -> BoundRegisteredDishMask:
    mask = RegisteredDishMask(
        key=RegisteredDishMaskKey("omnifin0", "shadow", "arena_1", "2010093"),
        artifact_id="dishrim-2010093",
        source_observation_sha256="sha256:" + "a" * 64,
        registration_id="dailyreg-1",
        registration_sha256="sha256:" + "b" * 64,
        source_contract_sha256="sha256:" + "c" * 64,
        h5_scope_sha256=None,
        physical_inner_rim=CircleGeometry(320.25, 240.5, 200.0),
        valid_detection_gate=CircleGeometry(320.25, 240.5, 205.0),
        native_width_px=640,
        native_height_px=480,
        coordinate_space="camera_native_pixels",
        palette_space_id="source_camera_image_px",
        coordinate_profile_id="source_camera_image_px.top_left_y_down.v1",
        pixel_convention="continuous",
        origin="top_left",
        positive_x="right",
        positive_y="down",
        target_plane="dish_top_rim",
        gating_semantics="bounding_box_centroid_inside_valid_detection_region",
        materialized_asset_status=MaterializedAssetStatus.COMPLETE,
        citrus_registration_status=CitrusRegistrationStatus.MISSING,
        source_valid_until_utc="2026-07-23T04:00:00Z",
        producer_operator_accepted=True,
        producer_quality_flags=(),
        selected_daily_registration_applied_by_citrus=False,
        source_kind="palette_recovered_recording_geometry",
        source_location="/recording/raw/recording_geometry_recovery.json",
        producer_contract_linkage_status="operator_approved_recovery_receipt",
        recovery_receipt_sha256="sha256:" + "d" * 64,
        independent_fit_required_before_operational_use=True,
    )
    return BoundRegisteredDishMask(
        mask=mask,
        pixel_frame_record_ref=(
            "/analysis/coordinate_frames/source_camera/2010093/"
            "continuous@pixel_frame_authority"
        ),
        pixel_frame_record_sha256="e" * 64,
    )


def _recovery_binding() -> dict[str, object]:
    return {
        "receipt_schema_id": "palette.recording_geometry_recovery_receipt",
        "receipt_id": "recovery-1",
        "receipt_sha256": "sha256:" + "d" * 64,
        "authority": mod.RECOVERY_AUTHORITY,
        "reason": mod.RECOVERY_REASON,
        "target_h5_sha256": "sha256:" + "f" * 64,
        "target_session_uuid": "session-1",
        "h5_geometry_capture_status": "not_referenced",
        "producer_artifacts_mutated": False,
    }


def _plan(source_zarr: Path) -> mod.ArenaGeometryCandidatePlan:
    record = mod.build_acquisition_geometry_candidate_record(
        _bound_mask(),
        recovery_binding=_recovery_binding(),
    )
    digest = hashlib.sha256(strict_json_dumps(record).encode("utf-8")).hexdigest()
    candidate_id = f"arena-geometry-acquisition-{digest[:24]}"
    provenance = build_writer_run_provenance(
        command="test_arena_geometry_candidate",
        params={"candidate_record_sha256": digest},
        cwd=Path(__file__).resolve().parents[3],
    )
    return mod.ArenaGeometryCandidatePlan(
        source_zarr=source_zarr,
        receipt_path=source_zarr.parent / "receipt.json",
        receipt_sha256="sha256:" + "d" * 64,
        candidate_id=candidate_id,
        candidate_record_sha256=digest,
        candidate_record=record,
        run_name=candidate_id,
        target_run_path=(
            source_zarr / "analysis" / mod.CANDIDATE_RUNS_PARENT / candidate_id
        ),
        run_provenance=provenance,
    )


def test_candidate_record_keeps_physical_rim_gate_and_selection_distinct() -> None:
    record = mod.build_acquisition_geometry_candidate_record(
        _bound_mask(),
        recovery_binding=_recovery_binding(),
    )

    assert record["physical_inner_rim"]["geometry"]["radius_px"] == 200.0
    assert record["valid_detection_region"]["geometry"]["radius_px"] == 205.0
    assert record["valid_detection_region"]["additional_palette_tolerance_px"] == 0.0
    assert record["coordinate_binding"]["pixel_convention"] == "continuous"
    assert record["candidate_policy"] == {
        "publication_role": "candidate_only",
        "operationally_selected": False,
        "legacy_dish_mask_projection_written": False,
        "detection_gate_applied": False,
        "independent_palette_fit_required_before_operational_use": True,
    }


def test_candidate_record_rejects_added_tolerance_or_nonconcentric_gate() -> None:
    record = mod.build_acquisition_geometry_candidate_record(
        _bound_mask(),
        recovery_binding=_recovery_binding(),
    )
    record["valid_detection_region"]["additional_palette_tolerance_px"] = 1.0
    with pytest.raises(RecordingGeometryError, match="added Palette tolerance"):
        mod.validate_acquisition_geometry_candidate_record(record)

    record = mod.build_acquisition_geometry_candidate_record(
        _bound_mask(),
        recovery_binding=_recovery_binding(),
    )
    record["valid_detection_region"]["geometry"]["center_px"]["x"] += 1.0
    with pytest.raises(RecordingGeometryError, match="concentric"):
        mod.validate_acquisition_geometry_candidate_record(record)


def test_atomic_candidate_publication_never_sets_latest_or_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_zarr = tmp_path / "recording.zarr"
    zarr.open_group(str(source_zarr), mode="w", zarr_format=3).require_group("analysis")
    plan = _plan(source_zarr)
    plan.receipt_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        mod,
        "_record_from_receipt_and_zarr",
        lambda _source, _receipt: (
            SimpleNamespace(receipt_sha256=plan.receipt_sha256),
            plan.candidate_record,
            plan.candidate_record_sha256,
        ),
    )

    result = mod.publish_arena_geometry_candidate(
        plan,
        scratch_root=tmp_path / "scratch",
        copy_backend="python",
    )

    assert result["published"] is True
    root = open_zarr_root(source_zarr, mode="r")
    parent = root[f"analysis/{mod.CANDIDATE_RUNS_PARENT}"]
    run = parent[plan.run_name]
    assert run.attrs["palette_run_completion_status"] == "complete"
    assert run.attrs["stage_selector_eligible"] is True
    assert run.attrs["operational_selection_status"] == "not_selected"
    assert run.attrs["legacy_dish_mask_projection_written"] is False
    assert run.attrs["detection_gate_applied"] is False
    assert "latest" not in parent.attrs
    assert "latest_complete" not in parent.attrs

    repeated = mod.publish_arena_geometry_candidate(
        plan,
        scratch_root=tmp_path / "scratch",
        copy_backend="python",
    )
    assert repeated["published"] is False
    assert repeated["status"] == "already_complete"
