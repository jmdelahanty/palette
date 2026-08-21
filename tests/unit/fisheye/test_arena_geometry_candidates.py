from __future__ import annotations

import hashlib
import json
from dataclasses import replace
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
    MaskGeometryStatus,
    MaterializedAssetStatus,
    RecordingGeometryError,
    RegisteredDishMask,
    RegisteredDishMaskCollection,
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


def _producer_mask(*, source_kind: str, source_location: str) -> RegisteredDishMask:
    return replace(
        _bound_mask().mask,
        source_kind=source_kind,
        source_location=source_location,
        producer_contract_linkage_status="producer_native",
        recovery_receipt_sha256=None,
    )


def _producer_collection(mask: RegisteredDishMask) -> RegisteredDishMaskCollection:
    return RegisteredDishMaskCollection(
        masks={mask.key: mask},
        mask_geometry_status=MaskGeometryStatus.VALID,
        source_kind=mask.source_kind,
        source_location=mask.source_location,
        source_contract_sha256=mask.source_contract_sha256,
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


def _palette_fit_inputs(tmp_path: Path) -> tuple[Path, Path]:
    fit = {
        "schema_id": "palette.diagnostics.recording_dish_rim_probe",
        "schema_version": 1,
        "status": "provisional_visual_review_required",
        "fit_frozen_before_acquisition_reveal": True,
        "fit_method": (
            "temporal_median_keyframe_only_multicandidate_radial_edge_circle_v2"
        ),
        "target_feature": "dish_inner_rim_water_side_edge",
        "parameters": {"acquisition_geometry_available_to_fitter": False},
        "source": {
            "camera_serial": "2010093",
            "image_shape_px": {"width": 640, "height": 480},
            "pixel_contract": "orange.camera.mono8.full_frame.v1",
            "video_path": "/recording/cams/Cam2010093_test.mp4",
            "video_size_bytes": 1234,
            "frame_count": 1000,
            "summary_sha256": "1" * 64,
        },
        "consensus_fit": {
            "coordinate_space": "camera_native_pixels",
            "geometry": {
                "type": "circle",
                "center_px": {"x": 320.0, "y": 240.0},
                "radius_px": 210.0,
            },
        },
        "fit_evidence_contract": {
            "all_window_candidates_frozen": True,
            "candidate_geometry_revealed_to_acquisition_fit": False,
            "candidate_feature_classification": "unclassified_concentric_rim_edge",
            "selection_scope": "window_consensus_for_review_not_operational_selection",
        },
        "windows": {},
    }
    for index, name in enumerate(("early", "middle", "late")):
        geometry = {
            "type": "circle",
            "center_px": {"x": 320.0 + index * 0.2, "y": 240.0},
            "radius_px": 210.0 + index * 0.1,
        }
        frame_indices = [
            99 + index * 300,
            100 + index * 300,
            101 + index * 300,
        ]
        fit["windows"][name] = {
            "center_frame": 100 + index * 300,
            "frame_indices": frame_indices,
            "decoded_luma_sequence_sha256": str(index + 2) * 64,
            "decoded_frames": [
                {
                    "frame_index": frame_index,
                    "decoded_frame_sha256": str(index + 7) * 64,
                }
                for frame_index in frame_indices
            ],
            "composite_pixel_sha256": str(index + 5) * 64,
            "fit": {
                "geometry": geometry,
                "angular_support_fraction": 0.98,
                "median_radial_gradient": 700.0,
                "radial_residual_px": 0.25,
                "selected_candidate_id": "candidate_000",
                "selection_reason": "highest_frozen_radial_evidence_score_v1",
                "frozen_candidates": [
                    {
                        "candidate_id": "candidate_000",
                        "geometry": geometry,
                        "coordinate_space": "camera_native_pixels",
                        "observed_feature_classification": (
                            "unclassified_concentric_rim_edge"
                        ),
                        "angular_support_fraction": 0.98,
                        "radial_residual_px": 0.25,
                        "median_radial_gradient": 700.0,
                        "evidence_score": 861.0,
                    }
                ],
            },
        }
    report = tmp_path / "fit_report.json"
    report.write_text(json.dumps(fit), encoding="utf-8")
    montage = tmp_path / "review_montage.png"
    montage.write_bytes(b"deterministic-review-montage")
    return report, montage


def _palette_binding() -> tuple[dict[str, object], dict[str, str], dict[str, object]]:
    coordinate = {
        "space_id": "source_camera_image_px",
        "profile_id": "source_camera_image_px.top_left_y_down.v1",
        "pixel_convention": "continuous",
        "units": "px",
        "origin": "top_left",
        "positive_x": "right",
        "positive_y": "down",
        "native_width_px": 640,
        "native_height_px": 480,
        "pixel_frame_record_ref": (
            "/analysis/coordinate_frames/source_camera/2010093/"
            "continuous@pixel_frame_authority"
        ),
        "pixel_frame_record_sha256": "e" * 64,
    }
    arena = {
        "rig_id": "omnifin0",
        "canvas_name": "shadow",
        "arena_id": "arena_1",
        "camera_serial": "2010093",
    }
    source = {
        "total_frames": 1000,
        "source_video": "Cam2010093_test.mp4",
        "file_fingerprint": {"size_bytes": 1234},
    }
    return coordinate, arena, source


def _clipped_palette_fit_inputs(
    recording: Path,
) -> tuple[Path, Path, Path, Path]:
    fit_report, montage = _palette_fit_inputs(recording)
    clip_index = recording / "recording_clip_index.json"
    clip_index.write_text('{"recording_id":"clipped-recording"}\n', encoding="utf-8")
    snapshot = (
        recording / "raw" / "recording_geometry_bundle" / "recording_snapshot.json"
    )
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text(
        json.dumps(
            {
                "camera_runtime": {
                    "2010093": {
                        "coordinate_frame": {
                            "coordinate_space": "camera_native_pixels",
                            "units": "pixels",
                            "origin": {"name": "top_left_pixel"},
                            "extent": {"width_px": 640, "height_px": 480},
                        }
                    }
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    report = json.loads(fit_report.read_text(encoding="utf-8"))
    report["source"].update(
        {
            "mode": "clipped_recording",
            "recording_dir": str(recording),
            "recording_id": "clipped-recording",
            "session_id": "clipped-session",
            "recording_clip_index_path": str(clip_index),
            "recording_clip_index_sha256": hashlib.sha256(
                clip_index.read_bytes()
            ).hexdigest(),
            "recording_geometry_snapshot_path": str(snapshot),
            "recording_geometry_snapshot_sha256": hashlib.sha256(
                snapshot.read_bytes()
            ).hexdigest(),
            "clip_count": 1,
            "sampled_clip_count": 1,
            "sampled_clips": [],
            "first_recording_frame_id": 1,
            "last_recording_frame_id": 1000,
            "source_binding": "test clipped collection",
        }
    )
    for window in report["windows"].values():
        frame_ids = [int(value) + 1 for value in window.pop("frame_indices")]
        center_id = int(window.pop("center_frame")) + 1
        window.update(
            {
                "center_recording_frame_id": center_id,
                "recording_frame_ids": frame_ids,
                "sampled_clip_ids": ["clip_000000"],
            }
        )
        window["decoded_frames"] = [
            {
                "clip_id": "clip_000000",
                "clip_index": 0,
                "clip_local_frame_index": frame_id - 1,
                "recording_frame_id": frame_id,
                "video_path": str(recording / "clips/clip_000000/camera.mp4"),
                "keyframe_path": str(
                    recording / "clips/clip_000000/camera_keyframe.json"
                ),
                "decoded_frame_sha256": decoded["decoded_frame_sha256"],
            }
            for frame_id, decoded in zip(frame_ids, window["decoded_frames"])
        ]
    fit_report.write_text(json.dumps(report), encoding="utf-8")
    return fit_report, montage, clip_index, snapshot


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


def test_reviewed_palette_candidate_binds_clipped_collection_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording = tmp_path / "recording"
    recording.mkdir()
    fit_report, montage, clip_index, snapshot = _clipped_palette_fit_inputs(recording)
    archive = recording / "zarr" / "recording_analysis.zarr"
    root = zarr.open_group(str(archive), mode="w", zarr_format=3)
    root.attrs.update(
        {
            "recording_id": "clipped-recording",
            "session_id": "clipped-session",
            "camera_serials": ["2010093"],
            "source_layout": "rolling_clips",
            "clip_count": 1,
            "recording_clip_index_json": str(clip_index),
            "recording_frame_index_row_count": 1000,
            "recording_frame_id_min": 1,
            "recording_frame_id_max": 1000,
        }
    )
    calibration = root.require_group("analysis/calibration")
    calibration.attrs.update(
        {
            "active_camera_id": "2010093",
            "native_width_px": 640,
            "native_height_px": 480,
        }
    )
    acquisition = SimpleNamespace(
        record=SimpleNamespace(camera_id="2010093"),
        record_ref=(
            "/analysis/acquisition_camera_frames/2010093"
            "@acquisition_camera_frame"
        ),
        record_sha256="f" * 64,
        width=640,
        height=480,
        assert_verified=lambda: None,
    )
    monkeypatch.setattr(
        mod,
        "load_persisted_acquisition_camera_authority",
        lambda *_args, **_kwargs: (object(), acquisition),
    )

    record = mod.build_reviewed_palette_geometry_candidate_record(
        source_zarr=archive,
        fit_report_path=fit_report,
        montage_path=montage,
        arena_binding={
            "rig_id": "omnifin0",
            "canvas_name": "shadow",
            "arena_id": "arena_1",
        },
        review={
            "status": "reviewer_accepted_for_offline_detection_gate_audit",
            "reviewer": "reviewer@example.org",
            "reviewed_at_utc": "2026-08-21T05:56:37Z",
            "decision_source": "interactive_visual_review",
            "reviewed_feature": "visible_dish_top_rim_edge",
            "decision_scope": "candidate_and_detection_disagreement_audit_only",
        },
    )

    coordinate = record["coordinate_binding"]
    assert coordinate["source_camera_frame_authority_kind"] == (
        mod.CLIPPED_ACQUISITION_FRAME_AUTHORITY_KIND
    )
    assert coordinate["pixel_frame_record_ref"] == (
        "/analysis/acquisition_camera_frames/2010093@acquisition_camera_frame"
    )
    assert coordinate["pixel_frame_record_sha256"] == "f" * 64
    source = record["palette_fit_source"]
    assert source["source_mode"] == "clipped_recording"
    assert source["source_collection"]["recording_id"] == "clipped-recording"
    assert source["source_collection"]["recording_geometry_snapshot_sha256"] == (
        hashlib.sha256(snapshot.read_bytes()).hexdigest()
    )
    assert source["windows"]["early"]["frame_coordinate"] == (
        "one_based_recording_frame_id"
    )

    legacy_record = json.loads(json.dumps(record))
    legacy_coordinate = legacy_record["coordinate_binding"]
    legacy_coordinate["source_camera_frame_authority_kind"] = (
        mod.LEGACY_CLIPPED_SNAPSHOT_FRAME_AUTHORITY_KIND
    )
    legacy_coordinate["pixel_frame_record_ref"] = (
        "/recording_geometry_snapshot/camera_runtime/2010093/"
        "coordinate_frame@recording_snapshot_sha256"
    )
    legacy_coordinate["pixel_frame_record_sha256"] = hashlib.sha256(
        snapshot.read_bytes()
    ).hexdigest()
    mod.validate_palette_geometry_candidate_record(legacy_record)

    acquisition.width = 641
    with pytest.raises(
        RecordingGeometryError,
        match="authority does not match the clipped recording snapshot",
    ):
        mod.build_reviewed_palette_geometry_candidate_record(
            source_zarr=archive,
            fit_report_path=fit_report,
            montage_path=montage,
            arena_binding={
                "rig_id": "omnifin0",
                "canvas_name": "shadow",
                "arena_id": "arena_1",
            },
            review=record["review"],
        )
    acquisition.width = 640

    clip_index.write_text('{"recording_id":"changed"}\n', encoding="utf-8")
    with pytest.raises(RecordingGeometryError, match="index changed"):
        mod.build_reviewed_palette_geometry_candidate_record(
            source_zarr=archive,
            fit_report_path=fit_report,
            montage_path=montage,
            arena_binding={
                "rig_id": "omnifin0",
                "canvas_name": "shadow",
                "arena_id": "arena_1",
            },
            review=record["review"],
        )


def test_plan_producer_native_folder_candidate_without_recovery_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording = tmp_path / "recording"
    source_zarr = recording / "zarr" / "recording_analysis.zarr"
    source_zarr.parent.mkdir(parents=True)
    contract = recording / "recording_geometry_contract.json"
    contract.write_text('{"contract":"exact"}\n', encoding="utf-8")
    contract_sha = hashlib.sha256(contract.read_bytes()).hexdigest()
    (recording / "recording_snapshot.json").write_text(
        json.dumps(
            {
                "recording_geometry_contract": {
                    "relative_path": contract.name,
                    "sha256": "sha256:" + contract_sha,
                }
            }
        ),
        encoding="utf-8",
    )
    mask = _producer_mask(
        source_kind="orange_recording_folder",
        source_location=str(recording.resolve()),
    )
    monkeypatch.setattr(
        mod,
        "load_registered_dish_masks_from_recording_folder",
        lambda *_args, **_kwargs: _producer_collection(mask),
    )
    monkeypatch.setattr(
        mod,
        "_bind_mask_to_zarr",
        lambda _source, selected: replace(_bound_mask(), mask=selected),
    )

    plan = mod.plan_producer_native_acquisition_geometry_candidate(
        source_zarr=source_zarr,
        recording_folder=recording,
        camera_serial="2010093",
        arena_id="arena_1",
    )

    source = plan.candidate_record["acquisition_source"]
    assert source["source_kind"] == "orange_recording_folder"
    assert source["recovery_binding"] is None
    assert plan.receipt_path == recording.resolve()
    assert [row["role"] for row in plan.run_provenance["input_artifacts"]] == [
        "recording_snapshot_geometry_pointer",
        "orange_recording_geometry_contract",
    ]


def test_plan_producer_native_folder_uses_fixed_organized_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording = tmp_path / "recording"
    source_zarr = recording / "zarr" / "recording_analysis.zarr"
    source_zarr.parent.mkdir(parents=True)
    bundle = recording / "raw" / "recording_geometry_bundle"
    bundle.mkdir(parents=True)
    contract = bundle / "recording_geometry_contract.json"
    contract.write_text('{"contract":"exact"}\n', encoding="utf-8")
    contract_sha = hashlib.sha256(contract.read_bytes()).hexdigest()
    (bundle / "recording_snapshot.json").write_text(
        json.dumps(
            {
                "recording_geometry_contract": {
                    "relative_path": contract.name,
                    "sha256": "sha256:" + contract_sha,
                }
            }
        ),
        encoding="utf-8",
    )
    mask = _producer_mask(
        source_kind="orange_recording_folder",
        source_location=str(bundle.resolve()),
    )
    observed: list[Path] = []

    def load_folder(path: Path, **_kwargs):
        observed.append(Path(path))
        return _producer_collection(mask)

    monkeypatch.setattr(
        mod,
        "load_registered_dish_masks_from_recording_folder",
        load_folder,
    )
    monkeypatch.setattr(
        mod,
        "_bind_mask_to_zarr",
        lambda _source, selected: replace(_bound_mask(), mask=selected),
    )

    plan = mod.plan_producer_native_acquisition_geometry_candidate(
        source_zarr=source_zarr,
        recording_folder=recording,
        camera_serial="2010093",
        arena_id="arena_1",
    )

    assert observed == [bundle.resolve()]
    assert plan.receipt_path == bundle.resolve()
    assert [
        Path(row["path"]).parent for row in plan.run_provenance["input_artifacts"]
    ] == [
        bundle.resolve(),
        bundle.resolve(),
    ]


def test_plan_producer_native_h5_requires_exact_camera_and_arena(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording = tmp_path / "recording"
    source_zarr = recording / "zarr" / "recording_analysis.zarr"
    source_zarr.parent.mkdir(parents=True)
    source_h5 = recording / "raw" / "session.h5"
    source_h5.parent.mkdir(parents=True)
    source_h5.write_bytes(b"exact-h5-fixture")
    mask = _producer_mask(
        source_kind="citrus_h5",
        source_location=str(source_h5.resolve()),
    )
    monkeypatch.setattr(
        mod,
        "load_registered_dish_masks_from_citrus_h5",
        lambda *_args, **_kwargs: _producer_collection(mask),
    )
    monkeypatch.setattr(
        mod,
        "_bind_mask_to_zarr",
        lambda _source, selected: replace(_bound_mask(), mask=selected),
    )

    with pytest.raises(RecordingGeometryError, match="exactly one requested"):
        mod.plan_producer_native_acquisition_geometry_candidate(
            source_zarr=source_zarr,
            citrus_h5=source_h5,
            camera_serial="2010094",
            arena_id="arena_1",
        )

    plan = mod.plan_producer_native_acquisition_geometry_candidate(
        source_zarr=source_zarr,
        citrus_h5=source_h5,
        camera_serial="2010093",
        arena_id="arena_1",
    )
    assert plan.candidate_record["acquisition_source"]["source_kind"] == "citrus_h5"
    assert plan.run_provenance["input_artifacts"][0]["role"] == (
        "citrus_h5_recording_geometry_contract"
    )


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


def test_reviewed_palette_candidate_corrects_semantics_and_keeps_gate_pointerless(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, montage = _palette_fit_inputs(tmp_path)
    monkeypatch.setattr(
        mod,
        "_source_camera_candidate_binding",
        lambda *_args, **_kwargs: _palette_binding(),
    )

    record = mod.build_reviewed_palette_geometry_candidate_record(
        source_zarr=tmp_path / "recording.zarr",
        fit_report_path=report,
        montage_path=montage,
        review={
            "status": "reviewer_accepted_for_offline_detection_gate_audit",
            "reviewer": "delahantyj",
            "reviewed_at_utc": "2026-07-26T23:40:00Z",
            "decision_source": "interactive_visual_review",
            "reviewed_feature": "visible_dish_top_rim_edge",
            "decision_scope": "candidate_and_detection_disagreement_audit_only",
        },
    )

    assert record["candidate_kind"] == mod.PALETTE_CANDIDATE_KIND
    assert (
        record["observed_boundary"]["observed_feature"] == "visible_dish_top_rim_edge"
    )
    early = record["palette_fit_source"]["windows"]["early"]
    assert early["selected_candidate_id"] == "candidate_000"
    assert early["radial_residual_px"] == pytest.approx(0.25)
    assert len(early["frozen_candidates"]) == 1
    assert [row["frame_index"] for row in early["decoded_frames"]] == [99, 100, 101]
    assert (
        record["palette_fit_source"]["probe_declared_target_feature"]
        == "dish_inner_rim_water_side_edge"
    )
    assert record["palette_fit_source"]["reviewed_semantic_correction"]["status"] == (
        "reviewer_corrected_probe_feature_label"
    )
    assert (
        record["valid_detection_region"]["geometry"]
        == record["observed_boundary"]["geometry"]
    )
    assert record["valid_detection_region"]["additional_palette_tolerance_px"] == 0.0
    assert record["candidate_policy"]["operationally_selected"] is False
    assert record["candidate_policy"]["detection_gate_applied"] is False
    assert record["palette_fit_source"]["acquisition_boundary_edge_support"] == {
        "status": "not_measured",
        "reason": "acquisition_reveal_not_present_at_candidate_publication",
    }


def test_palette_candidate_binds_post_freeze_acquisition_edge_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, montage = _palette_fit_inputs(tmp_path)
    geometry = {
        "type": "circle",
        "center_px": {"x": 320.0, "y": 240.0},
        "radius_px": 205.0,
    }
    window_support = {
        "status": "measured",
        "method": "fixed_circle_radial_gradient_support_v1",
        "geometry_frozen": True,
        "radial_band_px": 4.0,
        "angular_sample_count": 1440,
        "angular_edge_support_fraction": 0.92,
        "median_radial_gradient": 650.0,
        "median_absolute_radial_offset_px": 0.4,
        "signed_median_radial_offset_px": 0.1,
    }
    reveal = {
        "schema_id": (
            "palette.diagnostics.recording_dish_rim_probe.acquisition_reveal"
        ),
        "schema_version": 1,
        "fit_report": {
            "path": report.name,
            "sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        },
        "acquisition_boundary_edge_support": {
            "status": "measured",
            "method": "fixed_circle_radial_gradient_support_v1",
            "fit_frozen_before_measurement": True,
            "coordinate_space": "camera_native_pixels",
            "geometry": geometry,
            "source_observation_sha256": "a" * 64,
            "windows": {name: window_support for name in ("early", "middle", "late")},
            "median_angular_edge_support_fraction": 0.92,
            "minimum_angular_edge_support_fraction": 0.92,
            "median_absolute_radial_offset_px": 0.4,
            "median_radial_gradient": 650.0,
        },
    }
    (tmp_path / "acquisition_reveal.json").write_text(
        json.dumps(reveal), encoding="utf-8"
    )
    monkeypatch.setattr(
        mod,
        "_source_camera_candidate_binding",
        lambda *_args, **_kwargs: _palette_binding(),
    )

    plan = mod.plan_reviewed_palette_geometry_candidate(
        source_zarr=tmp_path / "recording.zarr",
        fit_report_path=report,
        montage_path=montage,
        reviewer="delahantyj",
        reviewed_at_utc="2026-08-13T12:00:00Z",
    )

    support = plan.candidate_record["palette_fit_source"][
        "acquisition_boundary_edge_support"
    ]
    assert support["status"] == "measured"
    assert support["geometry"] == geometry
    assert plan.run_provenance["input_artifacts"][2]["role"] == (
        "post_freeze_acquisition_boundary_edge_support"
    )


def test_reviewed_palette_candidate_rejects_mutated_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, montage = _palette_fit_inputs(tmp_path)
    monkeypatch.setattr(
        mod,
        "_source_camera_candidate_binding",
        lambda *_args, **_kwargs: _palette_binding(),
    )
    record = mod.build_reviewed_palette_geometry_candidate_record(
        source_zarr=tmp_path / "recording.zarr",
        fit_report_path=report,
        montage_path=montage,
        review={
            "status": "reviewer_accepted_for_offline_detection_gate_audit",
            "reviewer": "delahantyj",
            "reviewed_at_utc": "2026-07-26T23:40:00Z",
            "decision_source": "interactive_visual_review",
            "reviewed_feature": "visible_dish_top_rim_edge",
            "decision_scope": "candidate_and_detection_disagreement_audit_only",
        },
    )
    record["valid_detection_region"]["geometry"]["radius_px"] += 1.0

    with pytest.raises(RecordingGeometryError, match="gate derivation"):
        mod.validate_palette_geometry_candidate_record(record)


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


def test_atomic_palette_candidate_publication_remains_audit_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_zarr = tmp_path / "recording.zarr"
    zarr.open_group(str(source_zarr), mode="w", zarr_format=3).require_group("analysis")
    report, montage = _palette_fit_inputs(tmp_path)
    monkeypatch.setattr(
        mod,
        "_source_camera_candidate_binding",
        lambda *_args, **_kwargs: _palette_binding(),
    )
    plan = mod.plan_reviewed_palette_geometry_candidate(
        source_zarr=source_zarr,
        fit_report_path=report,
        montage_path=montage,
        reviewer="delahantyj",
        reviewed_at_utc="2026-07-26T23:40:00Z",
    )
    monkeypatch.setattr(
        mod,
        "_revalidate_candidate_sources",
        lambda _plan: {
            "status": "current",
            "source_kind": mod.PALETTE_CANDIDATE_KIND,
        },
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
    assert run.attrs["candidate_kind"] == mod.PALETTE_CANDIDATE_KIND
    assert run.attrs["stage_selector_eligible"] is True
    assert run.attrs["operational_selection_status"] == "not_selected"
    assert run.attrs["detection_gate_applied"] is False
    assert "latest" not in parent.attrs
    assert "latest_complete" not in parent.attrs


def test_candidate_reread_preserves_creation_provenance_across_software_contexts(
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
    mod.publish_arena_geometry_candidate(
        plan,
        scratch_root=tmp_path / "scratch",
        copy_backend="python",
    )

    replay_provenance = dict(plan.run_provenance)
    replay_provenance.update(
        {
            "git_sha": "f" * 40,
            "git_short_sha": "f" * 8,
            "git_dirty": False,
            "fisheye_version": "99.0.0",
            "runtime": {"host": {"hostname": "future-validator"}},
        }
    )
    replay_plan = replace(plan, run_provenance=replay_provenance)

    validation = mod.validate_arena_geometry_candidate_run(
        plan.target_run_path,
        expected_plan=replay_plan,
        require_complete=True,
        require_eligible=True,
    )

    assert validation["valid"] is True
    assert validation["errors"] == []


def test_candidate_reread_rejects_changed_stable_provenance_inputs(
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
    mod.publish_arena_geometry_candidate(
        plan,
        scratch_root=tmp_path / "scratch",
        copy_backend="python",
    )

    replay_provenance = dict(plan.run_provenance)
    replay_provenance["input_artifacts"] = [
        {
            "role": "recording_geometry_recovery_receipt",
            "path": str(plan.receipt_path),
            "sha256": "sha256:" + "0" * 64,
        }
    ]
    replay_plan = replace(plan, run_provenance=replay_provenance)

    validation = mod.validate_arena_geometry_candidate_run(
        plan.target_run_path,
        expected_plan=replay_plan,
        require_complete=True,
        require_eligible=True,
    )

    assert validation["valid"] is False
    assert validation["errors"] == ["run provenance input_artifacts mismatch"]
