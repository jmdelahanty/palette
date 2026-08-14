from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import fisheye.registry.geometry_review_approval as approval
import fisheye.utils.apply_geometry_review_approval as apply_approval
import fisheye.utils.submit_geometry_review_approval as submit_approval
from fisheye.cluster.geometry_review_approval import (
    build_geometry_review_approval_workflow,
)
from fisheye.registry.geometry_review import (
    GeometryReviewQueueItem,
    GeometryStageState,
)
from fisheye.shared.zarr.detection_schema import (
    CanonicalDetectionDimensions,
    derive_canonical_detection_geometry,
)


class _Root:
    def __init__(self, *, attrs: dict[str, object] | None = None) -> None:
        self.attrs = dict(attrs or {})


class _Group(dict[str, object]):
    def __init__(self, values: dict[str, object], *, attrs: dict[str, object]) -> None:
        super().__init__(values)
        self.attrs = attrs


def _modern_detection_group() -> _Group:
    instance_keys = np.asarray([11, 12, 13], dtype=np.uint64)
    source_frames = np.asarray([0, 2, 4], dtype=np.int64)
    row_count = int(instance_keys.shape[0])
    bbox_norm = np.tile(
        np.asarray([[0.5, 0.5, 0.1, 0.1]], dtype=np.float32),
        (row_count, 1),
    )
    bbox_img, centers_img = derive_canonical_detection_geometry(
        bbox_norm,
        source_width=4512,
        source_height=4512,
    )
    group_path = "detect_runs/raw-modern"
    temporal = {
        "schema_id": "palette.source_row_temporal_authority",
        "schema_version": 1,
        "acquisition_camera_frame": {
            "record_ref": "/analysis/acquisition_camera_frames/camera-1@acquisition_camera_frame",
            "record_sha256": "a" * 64,
        },
        "recording_id": "recording-1",
        "camera_id": "camera-1",
        "source_total_frames": 5,
        "source_rowset_ref": f"/{group_path}",
        "source_row_identity": {
            "record_ref": f"/{group_path}@row_identity_contract",
            "record_sha256": "b" * 64,
        },
        "source_identity_domain": "observation_instance",
        "source_identity_mode": "instance_key",
        "source_leading_dimension": row_count,
        "source_acquisition_frame_index": {
            "ref": f"/{group_path}/instances/source_acquisition_frame_index",
            "dtype": "<i8",
            "shape": [row_count],
            "content_sha256": approval.array_values_sha256(source_frames),
        },
        "observation_instance_key": {
            "ref": f"/{group_path}/instances/instance_key",
            "dtype": "<u8",
            "shape": [row_count],
            "content_sha256": approval.array_values_sha256(instance_keys),
        },
    }
    attrs = {
        "palette_run_completion_status": "complete",
        "run_manifest": {
            "schema_id": "palette.canonical_detection.run_manifest",
            "schema_version": 3,
            "payload_digest": "e" * 64,
            "payload": {"run_id": "raw-modern"},
        },
        "validated_backend_result_count": 5,
        "source_row_temporal_authority": temporal,
        "source_row_temporal_authority_sha256": approval._sha256(temporal),
        "source_pixel_authority": {
            "record_ref": "/analysis/coordinate_frames/source_camera/camera-1/continuous@pixel_frame_authority",
            "record_sha256": "c" * 64,
        },
        "immutable_yolo_storage_validation": {
            "schema_id": "palette.immutable_yolo_storage_completion.v1",
            "status": "ok",
            "stage": "detect",
            "row_count": row_count,
            "frame_count": 5,
            "errors": [],
        },
    }
    instances = _Group(
        {
            "instance_key": instance_keys,
            "bbox_norm_coords": bbox_norm,
            "bbox_img_xyxy": bbox_img,
            "centers_img_xy": centers_img,
            "scores": np.ones(row_count, dtype=np.float32),
            "class_ids": np.zeros(row_count, dtype=np.int32),
            "frame_indices": np.asarray([0, 2, 4], dtype=np.int32),
            "source_acquisition_frame_index": source_frames,
            "frame_row_offsets": np.asarray([0, 1, 1, 2, 2, 3], dtype=np.int64),
        },
        attrs={},
    )
    return _Group(
        {"instances": instances},
        attrs=attrs,
    )


def _stub_canonical_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        approval,
        "canonical_detection_dimensions_from_manifest",
        lambda _manifest: CanonicalDetectionDimensions(
            n_frames=5,
            n_instances=3,
            source_width=4512,
            source_height=4512,
        ),
    )


def _build_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    selected: str = "palette",
) -> approval.GeometryReviewApprovalRequest:
    zarr_path = tmp_path / "recording_analysis.zarr"
    zarr_path.mkdir(exist_ok=True)
    recording_dir = tmp_path / "recording"
    recording_dir.mkdir(exist_ok=True)
    video = recording_dir / "camera.mp4"
    video.touch(exist_ok=True)
    root = _Root(
        attrs={
            "recording_id": "recording-1",
            "recording_path": str(recording_dir),
            "arena_id": "arena-1",
            "camera_id": "camera-1",
            "experiment_setup": {"expected_subject_count": 1},
        }
    )
    monkeypatch.setattr(approval, "open_zarr_root", lambda *_a, **_k: root)
    monkeypatch.setattr(
        approval,
        "_fit_review_binding",
        lambda _root, run: {
            "run_name": run,
            "review_record_sha256": "1" * 64,
            "camera_serial": "camera-1",
            "frame_count": 100,
            "video_path": str(video),
        },
    )
    monkeypatch.setattr(
        approval,
        "_candidate_binding",
        lambda _root, run: {
            "run_name": run,
            "candidate_kind": approval.ACQUISITION_CANDIDATE_KIND,
            "candidate_record_sha256": "2" * 64,
            "arena_binding": {"camera_serial": "camera-1"},
            "coordinate_binding": {
                "coordinate_space": "camera_native_pixels",
                "native_width_px": 4512,
                "native_height_px": 4512,
            },
        },
    )
    monkeypatch.setattr(
        approval,
        "detection_source_binding",
        lambda _root, path: {
            "group_path": path,
            "run_name": path.split("/")[-1],
            "row_count": 10,
            "frame_count": 100,
            "source_video_width": 4512,
            "source_video_height": 4512,
            "binding_sha256": "3" * 64,
        },
    )
    monkeypatch.setattr(
        approval,
        "plan_reviewed_palette_geometry_candidate",
        lambda **_kwargs: SimpleNamespace(
            candidate_id="arena-geometry-palette-reviewed",
            candidate_record_sha256="4" * 64,
        ),
    )
    return approval.build_geometry_review_approval_request(
        registry_path=tmp_path / "registry.sqlite",
        dataset_id="dataset-1",
        recording_id="recording-1",
        analysis_zarr=zarr_path,
        fit_review_run="fit-review-1",
        acquisition_candidate_run="acquisition-1",
        source_detection_group_path="detect_runs/raw-1",
        selected_candidate_kind=selected,
        semantic_compatibility="different_feature_confirmed",
        reviewer="operator@example.org",
        reviewed_at_utc="2026-08-14T12:00:00Z",
        decision_reason="Palette follows the visible operational boundary better.",
        palette_commit="a" * 40,
    )


def test_approval_request_is_content_addressed_and_freezes_exact_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _build_request(tmp_path, monkeypatch)
    second = _build_request(tmp_path, monkeypatch)

    assert first == second
    assert first.request_id.startswith("geometry_review_approval_")
    assert first.payload["identity"]["detection_source"] == {
        "binding_sha256": "3" * 64,
        "group_path": "detect_runs/raw-1",
        "row_count": 10,
        "run_name": "raw-1",
        "frame_count": 100,
        "source_video_width": 4512,
        "source_video_height": 4512,
    }
    assert first.payload["identity"]["decision"]["selected_candidate_kind"] == (
        "palette"
    )
    assert first.payload["pipeline"]["registered_gate_requirement"] == "required"
    assert approval.validate_geometry_review_approval_request(first.payload) == first


def test_approval_request_changes_when_operator_choice_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    palette = _build_request(tmp_path, monkeypatch, selected="palette")
    acquisition = _build_request(tmp_path, monkeypatch, selected="acquisition")

    assert palette.request_id != acquisition.request_id
    assert palette.gate_run != acquisition.gate_run


def test_detection_binding_uses_sealed_modern_frame_count_authorities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = "detect_runs/raw-modern"
    group = _modern_detection_group()
    _stub_canonical_dimensions(monkeypatch)

    binding = approval.detection_source_binding({path: group}, path)

    assert binding["frame_count"] == 5
    assert binding["frame_count_authority"]["declarations"] == {
        "canonical_run_manifest": 5,
        "source_row_temporal_authority": 5,
        "immutable_yolo_storage_validation": 5,
        "validated_backend_result_count": 5,
        "array:frame_row_offsets": 5,
    }
    assert binding["frame_count_authority"][
        "source_row_temporal_authority_sha256"
    ] == group.attrs["source_row_temporal_authority_sha256"]


def test_detection_binding_rejects_disagreeing_modern_frame_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = "detect_runs/raw-modern"
    group = _modern_detection_group()
    group.attrs["validated_backend_result_count"] = 6
    _stub_canonical_dimensions(monkeypatch)

    with pytest.raises(
        approval.GeometryReviewApprovalError,
        match="frame-count authorities disagree",
    ):
        approval.detection_source_binding({path: group}, path)


def test_detection_binding_rejects_stale_temporal_authority_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = "detect_runs/raw-modern"
    group = _modern_detection_group()
    group.attrs["source_row_temporal_authority_sha256"] = "d" * 64
    _stub_canonical_dimensions(monkeypatch)

    with pytest.raises(
        approval.GeometryReviewApprovalError,
        match="temporal-authority digest is stale",
    ):
        approval.detection_source_binding({path: group}, path)


def test_detection_binding_rejects_flat_run_without_manifest() -> None:
    path = "detect_runs/raw-modern"
    group = _modern_detection_group()
    group.attrs.pop("run_manifest")

    with pytest.raises(
        approval.GeometryReviewApprovalError,
        match="lacks its exact canonical run_manifest",
    ):
        approval.detection_source_binding({path: group}, path)


def test_detection_binding_rejects_pre_coordinate_manifest() -> None:
    path = "detect_runs/raw-modern"
    group = _modern_detection_group()
    group.attrs["run_manifest"]["schema_version"] = 2

    with pytest.raises(
        approval.GeometryReviewApprovalError,
        match="not a coordinate-aware canonical-v3 run",
    ):
        approval.detection_source_binding({path: group}, path)


def test_registry_precondition_requires_exact_actionable_pending_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _build_request(tmp_path, monkeypatch)
    fit_stage = GeometryStageState(
        step_name="arena_geometry_offline_fit",
        status="ok",
        run_name="fit-review-1",
        review_status={
            "state": "evidence_complete_review_pending",
            "runs": ["fit-review-1"],
        },
        details=None,
        source="test",
        updated_utc="2026-08-14T12:00:00Z",
    )
    item = GeometryReviewQueueItem(
        dataset_id="dataset-1",
        recording_id="recording-1",
        zarr_path=tmp_path / "recording_analysis.zarr",
        camera_serial="camera-1",
        arena_id="arena-1",
        geometry_state="fit_evidence_awaiting_review",
        actionable=True,
        stages=(fit_stage,),
    )
    monkeypatch.setattr(
        approval, "load_geometry_review_queue", lambda *_a, **_k: [item]
    )

    approval.verify_geometry_review_registry_precondition(request)

    monkeypatch.setattr(
        approval,
        "load_geometry_review_queue",
        lambda *_a, **_k: [
            GeometryReviewQueueItem(**{**item.__dict__, "actionable": False})
        ],
    )
    with pytest.raises(approval.GeometryReviewApprovalError, match="non-actionable"):
        approval.verify_geometry_review_registry_precondition(request)


def test_approval_workflow_is_commit_pinned_and_dependency_ordered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _build_request(tmp_path, monkeypatch)
    plan = build_geometry_review_approval_workflow(
        request,
        request_path=tmp_path / "request.json",
        palette_repo=tmp_path / "palette-deployment",
        run_root=tmp_path / "operations" / request.request_id,
    )

    ordered = plan.workflow.topological_jobs()
    assert [job.metadata["stage"] for job in ordered] == [
        "geometry_review_approval_publication",
        "detect_quality",
        "detect_refine",
        "geometry_review_registry_refresh",
    ]
    assert ordered[0].resources.ncores == 1
    assert ordered[0].resources.mem_gb == 8
    assert "--apply" in ordered[0].command
    refine_command = " ".join(ordered[2].command)
    registry_command = " ".join(ordered[-1].command)
    assert "--registered-gate-requirement" in refine_command
    assert "required" in refine_command
    assert "--reconcile-step-status" in registry_command
    assert "--safe-shadow-publish" in registry_command
    assert "--backup-path" in registry_command
    assert str(plan.registry_backup_path) in registry_command
    assert plan.workflow.metadata["palette_commit"] == "a" * 40
    assert plan.workflow.metadata["raw_detections_preserved"] is True
    assert plan.workflow.metadata["processing_scope"] == (
        "geometry_quality_refinement"
    )
    assert plan.workflow.metadata["crop_submission"] == "deferred"
    assert plan.workflow.metadata["registry_publication_mode"] == (
        "local_shadow_copy_atomic_replace"
    )


def test_submit_mode_requires_clean_ci_green_deployment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = {
        "registry_path": tmp_path / "registry.sqlite",
        "dataset_id": "dataset-1",
        "recording_id": "recording-1",
        "analysis_zarr": tmp_path / "recording_analysis.zarr",
        "fit_review_run": "fit-review-1",
        "acquisition_candidate_run": "acquisition-1",
        "source_detection_group_path": "detect_runs/raw-1",
        "selected_candidate_kind": "palette",
        "semantic_compatibility": "different_feature_confirmed",
        "reviewer": "operator",
        "reviewed_at_utc": "2026-08-14T12:00:00Z",
        "decision_reason": "reviewed",
        "palette_repo": tmp_path / "palette",
        "approval_root": tmp_path / "operations",
        "submit": True,
    }
    monkeypatch.setattr(submit_approval, "_git_state", lambda _repo: ("a" * 40, False))
    with pytest.raises(RuntimeError, match="clean Palette deployment"):
        submit_approval.prepare_geometry_review_approval_submission(
            **kwargs, required_ci_success=True
        )

    monkeypatch.setattr(submit_approval, "_git_state", lambda _repo: ("a" * 40, True))
    with pytest.raises(RuntimeError, match="required-CI"):
        submit_approval.prepare_geometry_review_approval_submission(
            **kwargs, required_ci_success=False
        )


def test_approval_state_cannot_use_campaign_staging_or_analysis_zarr(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "recording" / "analysis.zarr"

    with pytest.raises(ValueError, match="outside staging"):
        submit_approval._durable_approval_root(
            tmp_path / "staging" / "geometry",
            analysis_zarr=archive,
        )
    with pytest.raises(ValueError, match="outside the canonical analysis Zarr"):
        submit_approval._durable_approval_root(
            archive / "operations",
            analysis_zarr=archive,
        )


def test_apply_sequence_preserves_raw_detection_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _build_request(tmp_path, monkeypatch)
    palette_plan = SimpleNamespace(
        candidate_id="arena-geometry-palette-reviewed",
        candidate_record_sha256="4" * 64,
    )
    comparison_plan = SimpleNamespace(
        comparison_id="comparison-1",
        comparison_record_sha256="5" * 64,
    )
    selection_plan = SimpleNamespace(
        selection_id="selection-1",
        selection_record_sha256="6" * 64,
    )
    gate_plan = SimpleNamespace(output_run=request.gate_run)
    calls: list[str] = []

    monkeypatch.setattr(apply_approval, "_git_commit", lambda _repo: "a" * 40)
    monkeypatch.setattr(
        apply_approval, "verify_geometry_review_registry_precondition", lambda _r: None
    )
    monkeypatch.setattr(
        apply_approval,
        "revalidate_geometry_review_approval_sources",
        lambda _r: palette_plan,
    )
    monkeypatch.setattr(
        apply_approval, "_conflicting_selection", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        apply_approval,
        "publish_arena_geometry_candidate",
        lambda *_a, **_k: calls.append("candidate") or {"status": "complete"},
    )
    monkeypatch.setattr(
        apply_approval,
        "build_arena_geometry_comparison_plan",
        lambda *_a, **_k: comparison_plan,
    )
    monkeypatch.setattr(
        apply_approval,
        "publish_arena_geometry_comparison",
        lambda *_a, **_k: calls.append("comparison") or {"status": "complete"},
    )
    monkeypatch.setattr(
        apply_approval,
        "build_arena_geometry_selection_plan",
        lambda *_a, **_k: selection_plan,
    )
    monkeypatch.setattr(
        apply_approval,
        "publish_arena_geometry_selection",
        lambda *_a, **_k: calls.append("selection") or {"status": "complete"},
    )
    monkeypatch.setattr(
        apply_approval,
        "build_registered_detection_gate_plan",
        lambda *_a, **_k: gate_plan,
    )
    monkeypatch.setattr(
        apply_approval,
        "publish_registered_detection_gate",
        lambda *_a, **_k: calls.append("gate") or {"status": "complete"},
    )
    monkeypatch.setattr(apply_approval, "open_zarr_root", lambda *_a, **_k: _Root())
    monkeypatch.setattr(
        apply_approval,
        "detection_source_binding",
        lambda *_a, **_k: request.payload["identity"]["detection_source"],
    )

    result = apply_approval.apply_geometry_review_approval(
        request,
        palette_repo=tmp_path / "palette",
        scratch_root=tmp_path / "scratch",
        apply=True,
    )

    assert calls == ["candidate", "comparison", "selection", "gate"]
    assert result["status"] == "complete"
    assert result["raw_detections_mutated"] is False
    assert result["source_detection_binding_unchanged"] is True
