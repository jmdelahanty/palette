from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.analysis_workflows.materializers import arena_geometry_candidates
from fisheye.analysis_workflows.materializers import arena_geometry_comparison as comparison
from fisheye.analysis_workflows.materializers import registered_detection_gate as gate
from fisheye.analysis_workflows.materializers import arena_geometry_selection as selection
from fisheye.shared.json_safety import strict_json_dumps
from fisheye.shared.zarr.detection_schema import CanonicalDetectionDimensions
from fisheye.shared.zarr_io import open_zarr_root
from tests.unit.fisheye.test_arena_geometry_candidates import (
    _bound_mask,
    _palette_binding,
    _palette_fit_inputs,
    _recovery_binding,
)


def _write_candidate(parent: zarr.Group, record: dict[str, object]) -> str:
    digest = hashlib.sha256(strict_json_dumps(record).encode("utf-8")).hexdigest()
    kind = str(record["candidate_kind"]).replace("_geometry", "")
    name = f"arena-geometry-{kind}-{digest[:24]}"
    run = parent.create_group(name)
    run.attrs.update(
        {
            "candidate_id": name,
            "candidate_record": record,
            "candidate_record_sha256": digest,
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
            "operational_selection_status": "not_selected",
            "detection_gate_applied": False,
        }
    )
    return name


def _archive_with_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    acquisition_edge_support_digest: str | None = None,
) -> tuple[Path, str, str]:
    manifest_reader = lambda root, *, group_path, **_kwargs: dict(
        root[group_path].attrs["run_manifest"]
    )
    monkeypatch.setattr(
        comparison,
        "require_active_coordinate_canonical_detection",
        manifest_reader,
    )
    monkeypatch.setattr(
        gate,
        "require_active_coordinate_canonical_detection",
        manifest_reader,
    )
    dimensions = lambda _manifest: CanonicalDetectionDimensions(
        n_frames=3,
        n_instances=3,
        source_width=640,
        source_height=480,
    )
    monkeypatch.setattr(
        comparison,
        "canonical_detection_dimensions_from_manifest",
        dimensions,
    )
    monkeypatch.setattr(
        gate,
        "canonical_detection_dimensions_from_manifest",
        dimensions,
    )
    archive = tmp_path / "recording.zarr"
    root = zarr.open_group(str(archive), mode="w", zarr_format=3)
    parent = root.require_group("analysis").create_group(
        arena_geometry_candidates.CANDIDATE_RUNS_PARENT
    )
    acquisition_record = arena_geometry_candidates.build_acquisition_geometry_candidate_record(
        _bound_mask(), recovery_binding=_recovery_binding()
    )
    fit_report, montage = _palette_fit_inputs(tmp_path)
    if acquisition_edge_support_digest is not None:
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
                "path": fit_report.name,
                "sha256": hashlib.sha256(fit_report.read_bytes()).hexdigest(),
            },
            "acquisition_boundary_edge_support": {
                "status": "measured",
                "method": "fixed_circle_radial_gradient_support_v1",
                "fit_frozen_before_measurement": True,
                "coordinate_space": "camera_native_pixels",
                "geometry": {
                    "type": "circle",
                    "center_px": {"x": 320.25, "y": 240.5},
                    "radius_px": 200.0,
                },
                "source_observation_sha256": acquisition_edge_support_digest,
                "windows": {
                    name: window_support for name in ("early", "middle", "late")
                },
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
        arena_geometry_candidates,
        "_source_camera_candidate_binding",
        lambda *_args, **_kwargs: _palette_binding(),
    )
    palette_record = arena_geometry_candidates.build_reviewed_palette_geometry_candidate_record(
        source_zarr=archive,
        fit_report_path=fit_report,
        montage_path=montage,
        review={
            "status": "reviewer_accepted_for_offline_detection_gate_audit",
            "reviewer": "reviewer@example.org",
            "reviewed_at_utc": "2026-08-12T12:00:00Z",
            "decision_source": "interactive_visual_review",
            "reviewed_feature": "visible_dish_top_rim_edge",
            "decision_scope": "candidate_and_detection_disagreement_audit_only",
        },
    )
    return (
        archive,
        _write_candidate(parent, acquisition_record),
        _write_candidate(parent, palette_record),
    )


def _write_nested_detection_source(archive: Path) -> str:
    root = open_zarr_root(archive, mode="a")
    run = root.require_group("detect_runs").create_group("detect_native")
    run.attrs.update(
        {
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
            "source_video_width": 640,
            "source_video_height": 480,
            "num_frames": 3,
            "source_evidence": {
                "source_pixel_authority": {
                    "record_ref": (
                        "/analysis/coordinate_frames/source_camera/2010093/"
                        "continuous@pixel_frame_authority"
                    ),
                    "record_sha256": "e" * 64,
                }
            },
            "run_manifest": {
                "schema_id": "palette.canonical_detection.run_manifest",
                "schema_version": 3,
                "payload_digest": "f" * 64,
                "payload": {"run_id": "detect_native"},
            },
        }
    )
    instances = run.create_group("instances")
    boxes = instances.create_array(
        "bbox_norm_coords",
        data=np.asarray(
            [
                [320.0 / 640.0, 240.0 / 480.0, 0.1, 0.1],
                [525.0 / 640.0, 240.0 / 480.0, 0.1, 0.1],
                [530.0 / 640.0, 240.0 / 480.0, 0.1, 0.1],
            ],
            dtype=np.float32,
        ),
    )
    boxes.attrs["coordinate_descriptor"] = {
        "geometry_type": "bbox_cxcywh",
        "component_units": ["normalized"] * 4,
    }
    instances.create_array("frame_indices", data=np.asarray([0, 1, 2], dtype=np.int32))
    instances.create_array("instance_key", data=np.asarray([10, 20, 30], dtype=np.uint64))
    return "detect_runs/detect_native"


def _use_canonical_v2_detection_authority(archive: Path, source_path: str) -> None:
    root = open_zarr_root(archive, mode="a")
    run = root[source_path]
    del run.attrs["source_evidence"]
    run.attrs["bbox_center_derivation"] = {
        "schema_id": "palette.bbox_center_derivation",
        "schema_version": 2,
        "operation": "half_open_xyxy_edges_to_continuous_midpoint_v2",
        "destination_frame": {
            "record_ref": (
                "/analysis/coordinate_frames/source_camera/2010093/"
                "continuous@pixel_frame_authority"
            ),
            "record_sha256": "sha256:" + "e" * 64,
        },
    }


def test_unresolved_comparison_preserves_semantics_and_blocks_automatic_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive, acquisition, palette = _archive_with_candidates(tmp_path, monkeypatch)
    detect_source = _write_nested_detection_source(archive)

    plan = comparison.build_arena_geometry_comparison_plan(
        archive,
        acquisition_candidate_run=acquisition,
        palette_candidate_run=palette,
        semantic_compatibility="projected_edges_unresolved",
        policy_id=comparison.CORROBORATED_ACQUISITION_POLICY_ID,
        detect_source_group_path=detect_source,
    )

    record = plan.comparison_record
    assert record["observed_features"]["acquisition"] == (
        "dish_inner_rim_water_side_edge"
    )
    assert record["observed_features"]["semantic_compatibility"] == (
        "projected_edges_unresolved"
    )
    assert record["geometry"]["same_feature_physical_boundary_metrics"] is None
    assert record["policy"]["automatic_selection_promoted"] is False
    assert record["policy"]["remaining_canary_measurements"]
    assert record["decision"]["candidate_selected"] is False
    assert record["decision"]["workflow_action"] == "review"
    operational = record["operational_gate_disagreement"]
    assert operational["status"] == "measured"
    assert operational["row_count"] == 3
    assert operational["additional_palette_tolerance_px"] == 0.0


def test_comparison_uses_canonical_manifest_extent_over_legacy_attrs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive, acquisition, palette = _archive_with_candidates(tmp_path, monkeypatch)
    detect_source = _write_nested_detection_source(archive)
    root = open_zarr_root(archive, mode="a")
    source = root[detect_source]
    source.attrs["source_video_width"] = 64
    source.attrs["source_video_height"] = 48
    source.attrs["run_manifest"] = {
        "schema_id": "palette.canonical_detection.run_manifest",
        "payload_digest": "f" * 64,
    }
    del source["instances/bbox_norm_coords"].attrs["coordinate_descriptor"]
    monkeypatch.setattr(
        comparison,
        "canonical_detection_dimensions_from_manifest",
        lambda _manifest: CanonicalDetectionDimensions(
            n_frames=3,
            n_instances=3,
            source_width=640,
            source_height=480,
        ),
    )

    plan = comparison.build_arena_geometry_comparison_plan(
        archive,
        acquisition_candidate_run=acquisition,
        palette_candidate_run=palette,
        semantic_compatibility="projected_edges_unresolved",
        policy_id=comparison.MANUAL_REVIEW_POLICY_ID,
        detect_source_group_path=detect_source,
    )

    signature = plan.comparison_record["operational_gate_disagreement"][
        "source_signature_payload"
    ]
    assert signature["source_video_width"] == 640
    assert signature["source_video_height"] == 480
    assert signature["canonical_run_manifest_payload_digest"] == "f" * 64


def test_comparison_normalizes_equivalent_observation_digest_encodings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive, acquisition, palette = _archive_with_candidates(
        tmp_path,
        monkeypatch,
        acquisition_edge_support_digest="a" * 64,
    )

    plan = comparison.build_arena_geometry_comparison_plan(
        archive,
        acquisition_candidate_run=acquisition,
        palette_candidate_run=palette,
        semantic_compatibility="projected_edges_unresolved",
    )

    assert plan.comparison_record["acquisition_boundary_edge_support"]["status"] == (
        "measured"
    )


def test_comparison_rejects_different_observation_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive, acquisition, palette = _archive_with_candidates(
        tmp_path,
        monkeypatch,
        acquisition_edge_support_digest="b" * 64,
    )

    with pytest.raises(ValueError, match="does not bind the producer observation"):
        comparison.build_arena_geometry_comparison_plan(
            archive,
            acquisition_candidate_run=acquisition,
            palette_candidate_run=palette,
            semantic_compatibility="projected_edges_unresolved",
        )


def test_same_feature_metrics_require_reviewed_semantic_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive, acquisition, palette = _archive_with_candidates(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="requires explicit reviewed evidence"):
        comparison.build_arena_geometry_comparison_plan(
            archive,
            acquisition_candidate_run=acquisition,
            palette_candidate_run=palette,
            semantic_compatibility="same_feature_confirmed",
        )

    plan = comparison.build_arena_geometry_comparison_plan(
        archive,
        acquisition_candidate_run=acquisition,
        palette_candidate_run=palette,
        semantic_compatibility="same_feature_confirmed",
        semantic_review={
            "reviewer": "reviewer@example.org",
            "reviewed_at_utc": "2026-08-12T12:30:00Z",
            "evidence_reason": "same water-side edge visible in all windows",
        },
    )
    metrics = plan.comparison_record["geometry"][
        "same_feature_physical_boundary_metrics"
    ]
    assert metrics["signed_radius_difference_px"] == pytest.approx(10.0)
    assert metrics["absolute_radius_difference_px"] == pytest.approx(10.0)


def test_comparison_publication_is_immutable_and_selector_ineligible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive, acquisition, palette = _archive_with_candidates(tmp_path, monkeypatch)
    plan = comparison.build_arena_geometry_comparison_plan(
        archive,
        acquisition_candidate_run=acquisition,
        palette_candidate_run=palette,
        semantic_compatibility="projected_edges_unresolved",
    )

    result = comparison.publish_arena_geometry_comparison(
        plan,
        scratch_root=tmp_path / "scratch",
    )

    assert result["published"] is True
    root = open_zarr_root(archive, mode="r")
    parent = root[f"analysis/{comparison.COMPARISON_RUNS_PARENT}"]
    run = parent[plan.comparison_id]
    assert run.attrs["palette_run_completion_status"] == "complete"
    assert run.attrs["stage_selector_eligible"] is False
    assert run.attrs["candidate_selected"] is False
    assert "latest" not in parent.attrs
    assert "latest_complete" not in parent.attrs


def test_comparison_rejects_duplicate_detection_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive, acquisition, palette = _archive_with_candidates(tmp_path, monkeypatch)
    detect_source = _write_nested_detection_source(archive)
    root = open_zarr_root(archive, mode="a")
    root[f"{detect_source}/instances/instance_key"][:] = np.asarray(
        [10, 10, 30], dtype=np.uint64
    )

    with pytest.raises(ValueError, match="not unique"):
        comparison.build_arena_geometry_comparison_plan(
            archive,
            acquisition_candidate_run=acquisition,
            palette_candidate_run=palette,
            semantic_compatibility="projected_edges_unresolved",
            detect_source_group_path=detect_source,
        )


def test_comparison_rejects_detection_source_with_wrong_pixel_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive, acquisition, palette = _archive_with_candidates(tmp_path, monkeypatch)
    detect_source = _write_nested_detection_source(archive)
    root = open_zarr_root(archive, mode="a")
    evidence = dict(root[detect_source].attrs["source_evidence"])
    authority = dict(evidence["source_pixel_authority"])
    authority["record_sha256"] = "f" * 64
    evidence["source_pixel_authority"] = authority
    root[detect_source].attrs["source_evidence"] = evidence

    with pytest.raises(ValueError, match="exact persisted source-camera pixel authority"):
        comparison.build_arena_geometry_comparison_plan(
            archive,
            acquisition_candidate_run=acquisition,
            palette_candidate_run=palette,
            semantic_compatibility="projected_edges_unresolved",
            detect_source_group_path=detect_source,
        )


def test_comparison_accepts_canonical_v2_center_derivation_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive, acquisition, palette = _archive_with_candidates(tmp_path, monkeypatch)
    detect_source = _write_nested_detection_source(archive)
    _use_canonical_v2_detection_authority(archive, detect_source)

    plan = comparison.build_arena_geometry_comparison_plan(
        archive,
        acquisition_candidate_run=acquisition,
        palette_candidate_run=palette,
        semantic_compatibility="projected_edges_unresolved",
        detect_source_group_path=detect_source,
    )

    assert plan.comparison_record["operational_gate_disagreement"]["status"] == (
        "measured"
    )


def test_selection_binds_exact_comparison_and_keeps_boundary_roles_distinct(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive, acquisition, palette = _archive_with_candidates(tmp_path, monkeypatch)
    comparison_plan = comparison.build_arena_geometry_comparison_plan(
        archive,
        acquisition_candidate_run=acquisition,
        palette_candidate_run=palette,
        semantic_compatibility="projected_edges_unresolved",
    )
    comparison.publish_arena_geometry_comparison(
        comparison_plan,
        scratch_root=tmp_path / "comparison-scratch",
    )

    selected = selection.build_arena_geometry_selection_plan(
        archive,
        candidate_run=acquisition,
        comparison_run=comparison_plan.comparison_id,
        selected_by="reviewer@example.org",
        decision_reason="reviewed unresolved projection and operational evidence",
    )

    record = selected.selection_record
    assert record["schema_version"] == selection.SELECTION_RECORD_SCHEMA_VERSION
    assert record["selected_candidate"]["boundary_observation"]["role"] == (
        "producer_physical_inner_rim"
    )
    assert record["selected_candidate"]["observed_boundary"] is None
    binding = record["decision"]["comparison_binding"]
    assert binding["run_name"] == comparison_plan.comparison_id
    assert binding["comparison_record_sha256"] == (
        comparison_plan.comparison_record_sha256
    )

    failed_binding = dict(binding)
    failed_binding["workflow_action"] = "fail"
    monkeypatch.setattr(
        selection,
        "_comparison_snapshot",
        lambda *_args, **_kwargs: failed_binding,
    )
    with pytest.raises(ValueError, match="failed geometry comparison cannot be overridden"):
        selection.build_arena_geometry_selection_plan(
            archive,
            candidate_run=acquisition,
            comparison_run=comparison_plan.comparison_id,
            selected_by="reviewer@example.org",
            decision_reason="attempted override",
        )

    with pytest.raises(ValueError, match="promoted corroborated acquisition pass"):
        selection.build_arena_geometry_selection_plan(
            archive,
            candidate_run=acquisition,
            comparison_run=comparison_plan.comparison_id,
            selected_by="policy",
            decision_reason="automatic",
            decision_source="automatic_policy",
        )


def test_comparison_bound_selection_gates_nested_canonical_detection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fisheye.analysis_workflows.materializers import registered_detection_gate

    archive, acquisition, palette = _archive_with_candidates(tmp_path, monkeypatch)
    detect_source = _write_nested_detection_source(archive)
    _use_canonical_v2_detection_authority(archive, detect_source)
    comparison_plan = comparison.build_arena_geometry_comparison_plan(
        archive,
        acquisition_candidate_run=acquisition,
        palette_candidate_run=palette,
        semantic_compatibility="projected_edges_unresolved",
        detect_source_group_path=detect_source,
    )
    comparison.publish_arena_geometry_comparison(
        comparison_plan,
        scratch_root=tmp_path / "comparison-scratch",
    )
    selection_plan = selection.build_arena_geometry_selection_plan(
        archive,
        candidate_run=acquisition,
        comparison_run=comparison_plan.comparison_id,
        selected_by="reviewer@example.org",
        decision_reason="reviewed exact independent and operational evidence",
    )
    selection.publish_arena_geometry_selection(
        selection_plan,
        scratch_root=tmp_path / "selection-scratch",
    )

    gate_plan = registered_detection_gate.build_registered_detection_gate_plan(
        archive,
        source_group_path=detect_source,
        selection_run=selection_plan.selection_id,
        inner_rows=2,
        shard_rows=4,
    )
    result = registered_detection_gate.publish_registered_detection_gate(
        gate_plan,
        scratch_root=tmp_path / "gate-scratch",
    )

    assert result["published"] is True
    root = open_zarr_root(archive, mode="r")
    gate_run = root[
        f"analysis/{registered_detection_gate.GATE_RUNS_PARENT}/{gate_plan.output_run}"
    ]
    np.testing.assert_array_equal(gate_run["instance_key"][:], [10, 20, 30])
    assert gate_run.attrs["source_pixel_frame_record_sha256"] == "e" * 64
    assert gate_run.attrs["selection_record_schema_version"] == 2
    assert gate_run.attrs["comparison_run"] == comparison_plan.comparison_id
    consumed = registered_detection_gate.validate_registered_detection_gate_consumption(
        archive,
        source_group_path=detect_source,
        gate_run=gate_plan.output_run,
        expected_instance_keys=np.asarray([10, 20, 30], dtype=np.uint64),
        require_comparison_bound_selection=True,
    )
    assert consumed["comparison_run"] == comparison_plan.comparison_id
    assert consumed["comparison_policy_id"] == "manual_review_only_v1"


def test_modern_gate_rejects_wrong_pixel_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fisheye.analysis_workflows.materializers import registered_detection_gate

    archive, acquisition, palette = _archive_with_candidates(tmp_path, monkeypatch)
    detect_source = _write_nested_detection_source(archive)
    comparison_plan = comparison.build_arena_geometry_comparison_plan(
        archive,
        acquisition_candidate_run=acquisition,
        palette_candidate_run=palette,
        semantic_compatibility="projected_edges_unresolved",
    )
    comparison.publish_arena_geometry_comparison(
        comparison_plan,
        scratch_root=tmp_path / "comparison-scratch",
    )
    selection_plan = selection.build_arena_geometry_selection_plan(
        archive,
        candidate_run=acquisition,
        comparison_run=comparison_plan.comparison_id,
        selected_by="reviewer@example.org",
        decision_reason="reviewed",
    )
    selection.publish_arena_geometry_selection(
        selection_plan,
        scratch_root=tmp_path / "selection-scratch",
    )
    root = open_zarr_root(archive, mode="a")
    evidence = dict(root[detect_source].attrs["source_evidence"])
    evidence["source_pixel_authority"] = {
        "record_ref": evidence["source_pixel_authority"]["record_ref"],
        "record_sha256": "f" * 64,
    }
    root[detect_source].attrs["source_evidence"] = evidence

    with pytest.raises(ValueError, match="exact persisted source-camera pixel"):
        registered_detection_gate.build_registered_detection_gate_plan(
            archive,
            source_group_path=detect_source,
            selection_run=selection_plan.selection_id,
        )
