from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.analysis_workflows.materializers import (
    arena_geometry_candidates as candidates,
)
from fisheye.analysis_workflows.materializers import (
    arena_geometry_selection as selection,
)
from fisheye.analysis_workflows.materializers import registered_detection_gate as gate
from fisheye.shared.json_safety import strict_json_dumps
from fisheye.shared.recording_geometry import (
    BoundRegisteredDishMask,
    CircleGeometry,
    CitrusRegistrationStatus,
    MaterializedAssetStatus,
    RegisteredDishMask,
    RegisteredDishMaskKey,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr.detection_schema import CanonicalDetectionDimensions


@pytest.fixture(autouse=True)
def _canonical_detection_authority_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gate,
        "require_active_coordinate_canonical_detection",
        lambda root, *, group_path, **_kwargs: root[group_path].attrs["run_manifest"],
    )
    monkeypatch.setattr(
        gate,
        "canonical_detection_dimensions_from_manifest",
        lambda _manifest: CanonicalDetectionDimensions(
            n_frames=10,
            n_instances=3,
            source_width=640,
            source_height=480,
        ),
    )


def _candidate_record() -> dict[str, object]:
    mask = RegisteredDishMask(
        key=RegisteredDishMaskKey("omnifin0", "shadow", "arena_1", "2010093"),
        artifact_id="dishrim-2010093",
        source_observation_sha256="sha256:" + "a" * 64,
        registration_id="dailyreg-1",
        registration_sha256="sha256:" + "b" * 64,
        source_contract_sha256="sha256:" + "c" * 64,
        h5_scope_sha256=None,
        physical_inner_rim=CircleGeometry(320.0, 240.0, 200.0),
        valid_detection_gate=CircleGeometry(320.0, 240.0, 205.0),
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
        source_valid_until_utc=None,
        producer_operator_accepted=True,
        producer_quality_flags=(),
        selected_daily_registration_applied_by_citrus=False,
        source_kind="orange_registered_observation",
        source_location="recording_geometry_contract.json",
        producer_contract_linkage_status="recording_bound",
        recovery_receipt_sha256=None,
        independent_fit_required_before_operational_use=True,
    )
    bound = BoundRegisteredDishMask(
        mask=mask,
        pixel_frame_record_ref=(
            "/analysis/coordinate_frames/source_camera/2010093/"
            "continuous@pixel_frame_authority"
        ),
        pixel_frame_record_sha256="e" * 64,
    )
    return candidates.build_acquisition_geometry_candidate_record(
        bound,
        recovery_binding=None,
    )


def _archive_with_candidate(tmp_path: Path) -> tuple[Path, str]:
    archive = tmp_path / "recording.zarr"
    root = zarr.open_group(str(archive), mode="w", zarr_format=3)
    analysis = root.create_group("analysis")
    parent = analysis.create_group(candidates.CANDIDATE_RUNS_PARENT)
    record = _candidate_record()
    digest = hashlib.sha256(strict_json_dumps(record).encode()).hexdigest()
    name = f"arena-geometry-acquisition-{digest[:24]}"
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
    return archive, name


def _publish_selection(
    archive: Path, candidate_run: str, tmp_path: Path
) -> selection.ArenaGeometrySelectionPlan:
    plan = selection.build_arena_geometry_selection_plan(
        archive,
        candidate_run=candidate_run,
        selected_by="reviewer@example.org",
        decision_reason="reviewed image fit supports this exact gate",
    )
    result = selection.publish_arena_geometry_selection(
        plan,
        scratch_root=tmp_path / "selection_scratch",
    )
    assert result["published"] is True
    return plan


def _write_detection_source(archive: Path) -> str:
    root = open_zarr_root(archive, mode="a")
    parent = root.create_group("detect_runs")
    source = parent.create_group("detect_test")
    source.attrs.update(
        {
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
            "source_video_width": 640,
            "source_video_height": 480,
            "num_frames": 10,
            "run_manifest": {
                "payload_digest": "f" * 64,
                "fixture": "canonical-v3",
            },
        }
    )
    bbox = source.create_array(
        "bbox_norm_coords",
        data=np.asarray(
            [
                [320.0 / 640.0, 240.0 / 480.0, 0.1, 0.1],
                [525.0 / 640.0, 240.0 / 480.0, 0.1, 0.1],
                [526.0 / 640.0, 240.0 / 480.0, 0.1, 0.1],
            ],
            dtype=np.float64,
        ),
    )
    bbox.attrs["coordinate_descriptor"] = {
        "geometry_type": "bbox_cxcywh",
        "component_units": ["normalized"] * 4,
    }
    source.create_array("frame_indices", data=np.asarray([0, 1, 2], dtype=np.int32))
    source.create_array("instance_key", data=np.asarray([10, 20, 30], dtype=np.uint64))
    return "detect_runs/detect_test"


def test_selection_is_separate_from_candidate_and_advances_guarded_pointer(
    tmp_path: Path,
) -> None:
    archive, candidate_run = _archive_with_candidate(tmp_path)
    plan = _publish_selection(archive, candidate_run, tmp_path)

    root = open_zarr_root(archive, mode="r")
    candidate = root[f"analysis/{candidates.CANDIDATE_RUNS_PARENT}/{candidate_run}"]
    parent = root[f"analysis/{selection.SELECTION_RUNS_PARENT}"]
    selected = parent[plan.selection_id]
    assert candidate.attrs["operational_selection_status"] == "not_selected"
    assert candidate.attrs["detection_gate_applied"] is False
    assert selected.attrs["stage_selector_eligible"] is True
    assert selected.attrs["operational_selection_status"] == "selected"
    assert parent.attrs["latest"] == plan.selection_id
    assert parent.attrs["latest_complete"] == plan.selection_id


def test_gate_publishes_keyed_sharded_decisions_without_mutating_raw_detection(
    tmp_path: Path,
) -> None:
    archive, candidate_run = _archive_with_candidate(tmp_path)
    selected = _publish_selection(archive, candidate_run, tmp_path)
    source_path = _write_detection_source(archive)
    plan = gate.build_registered_detection_gate_plan(
        archive,
        source_group_path=source_path,
        selection_run=selected.selection_id,
        inner_rows=2,
        shard_rows=4,
    )

    result = gate.publish_registered_detection_gate(
        plan,
        scratch_root=tmp_path / "gate_scratch",
    )

    assert result["published"] is True
    root = open_zarr_root(archive, mode="r")
    raw = root[source_path]
    run = root[f"analysis/{gate.GATE_RUNS_PARENT}/{plan.output_run}"]
    np.testing.assert_array_equal(run["instance_key"][:], [10, 20, 30])
    np.testing.assert_array_equal(run["source_row_index"][:], [0, 1, 2])
    np.testing.assert_array_equal(
        run["inside_registered_dish_mask"][:], [True, True, False]
    )
    np.testing.assert_array_equal(run["gate_decision"][:], [1, 1, 2])
    assert run.attrs["decision_summary"] == {
        "row_count": 3,
        "accepted_count": 2,
        "rejected_count": 1,
        "complete": True,
    }
    assert run.attrs["raw_detections_preserved"] is True
    assert set(raw.array_keys()) == {
        "bbox_norm_coords",
        "frame_indices",
        "instance_key",
    }
    assert root[f"analysis/{gate.GATE_RUNS_PARENT}"].attrs["latest"] == plan.output_run
    assert run["instance_key"].shards == (4,)

    with pytest.raises(ValueError, match="comparison-bound version-2 selection"):
        gate.validate_registered_detection_gate_consumption(
            archive,
            source_group_path=source_path,
            gate_run=plan.output_run,
            expected_instance_keys=np.asarray([10, 20, 30], dtype=np.uint64),
            require_comparison_bound_selection=True,
        )

    with pytest.raises(ValueError, match="modern operational selection"):
        gate.validate_registered_detection_gate_consumption(
            archive,
            source_group_path=source_path,
            gate_run=plan.output_run,
            expected_instance_keys=np.asarray([10, 20, 30], dtype=np.uint64),
            require_modern_operational_selection=True,
        )


def test_gate_can_revalidate_explicit_selector_ineligible_source_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    archive, candidate_run = _archive_with_candidate(tmp_path)
    selected = _publish_selection(archive, candidate_run, tmp_path)
    source_path = _write_detection_source(archive)
    root = open_zarr_root(archive, mode="a")
    source = root[source_path]
    source.attrs.update(
        {
            "stage_selector_eligible": False,
            "production_candidate": True,
            "run_manifest": {
                "schema_version": 3,
                "payload_digest": "f" * 64,
                "payload": {
                    "run_id": "detect_test",
                    "publication": {"stage_selector_eligible": False},
                },
            },
        }
    )
    monkeypatch.setattr(gate, "validate_canonical_detection_run_manifest", lambda _: ())
    monkeypatch.setattr(
        gate,
        "require_active_coordinate_canonical_detection",
        lambda *_a, **_k: (_ for _ in ()).throw(
            ValueError("candidate is intentionally not selected")
        ),
    )

    plan = gate.build_registered_detection_gate_plan(
        archive,
        source_group_path=source_path,
        selection_run=selected.selection_id,
        inner_rows=2,
        shard_rows=4,
        allow_selector_ineligible_source=True,
    )
    gate.publish_registered_detection_gate(
        plan,
        scratch_root=tmp_path / "candidate_gate_scratch",
    )
    consumed = gate.validate_registered_detection_gate_consumption(
        archive,
        source_group_path=source_path,
        gate_run=plan.output_run,
        expected_instance_keys=np.asarray([10, 20, 30], dtype=np.uint64),
        allow_selector_ineligible_source=True,
    )

    assert plan.allow_selector_ineligible_source is True
    assert consumed["source_detection_group_path"] == source_path
    np.testing.assert_array_equal(consumed["inside"], [True, True, False])


def test_gate_fails_closed_on_duplicate_modern_identity(tmp_path: Path) -> None:
    archive, candidate_run = _archive_with_candidate(tmp_path)
    selected = _publish_selection(archive, candidate_run, tmp_path)
    source_path = _write_detection_source(archive)
    root = open_zarr_root(archive, mode="a")
    root[source_path]["instance_key"][:] = np.asarray([10, 10, 30], dtype=np.uint64)
    plan = gate.build_registered_detection_gate_plan(
        archive,
        source_group_path=source_path,
        selection_run=selected.selection_id,
        inner_rows=2,
        shard_rows=4,
    )

    try:
        gate.publish_registered_detection_gate(
            plan,
            scratch_root=tmp_path / "gate_scratch",
        )
    except RuntimeError as exc:
        assert "instance_key values are not unique" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("Duplicate modern identity was accepted.")


def test_gate_validation_detects_later_source_payload_mutation(tmp_path: Path) -> None:
    archive, candidate_run = _archive_with_candidate(tmp_path)
    selected = _publish_selection(archive, candidate_run, tmp_path)
    source_path = _write_detection_source(archive)
    plan = gate.build_registered_detection_gate_plan(
        archive,
        source_group_path=source_path,
        selection_run=selected.selection_id,
        inner_rows=2,
        shard_rows=4,
    )
    gate.publish_registered_detection_gate(
        plan,
        scratch_root=tmp_path / "gate_scratch",
    )
    root = open_zarr_root(archive, mode="a")
    root[source_path]["bbox_norm_coords"][0, 0] = 0.25

    report = gate.validate_registered_detection_gate_run(
        plan.target_run_path,
        expected_plan=plan,
        require_complete=True,
        require_eligible=True,
    )

    assert report["valid"] is False
    assert "source detection payload changed" in report["errors"]


def test_gate_reads_extent_from_canonical_detection_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    archive, candidate_run = _archive_with_candidate(tmp_path)
    selected = _publish_selection(archive, candidate_run, tmp_path)
    source_path = _write_detection_source(archive)
    root = open_zarr_root(archive, mode="a")
    source = root[source_path]
    source.attrs["source_video_width"] = 64
    source.attrs["source_video_height"] = 48
    source.attrs["num_frames"] = 1
    del source["bbox_norm_coords"].attrs["coordinate_descriptor"]
    source.attrs["run_manifest"] = {
        "payload_digest": "f" * 64,
        "fixture": "canonical-v3",
    }
    monkeypatch.setattr(
        gate,
        "canonical_detection_dimensions_from_manifest",
        lambda manifest: CanonicalDetectionDimensions(
            n_frames=10,
            n_instances=3,
            source_width=640,
            source_height=480,
        ),
    )

    plan = gate.build_registered_detection_gate_plan(
        archive,
        source_group_path=source_path,
        selection_run=selected.selection_id,
        inner_rows=2,
        shard_rows=4,
    )

    assert (plan.width_px, plan.height_px, plan.frame_count) == (640, 480, 10)
    assert plan.source_signature


def test_gate_rejects_canonical_manifest_row_count_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    archive, candidate_run = _archive_with_candidate(tmp_path)
    selected = _publish_selection(archive, candidate_run, tmp_path)
    source_path = _write_detection_source(archive)
    root = open_zarr_root(archive, mode="a")
    root[source_path].attrs["run_manifest"] = {
        "payload_digest": "f" * 64,
        "fixture": "canonical-v3",
    }
    monkeypatch.setattr(
        gate,
        "canonical_detection_dimensions_from_manifest",
        lambda manifest: CanonicalDetectionDimensions(
            n_frames=10,
            n_instances=4,
            source_width=640,
            source_height=480,
        ),
    )

    with pytest.raises(ValueError, match="manifest instance count"):
        gate.build_registered_detection_gate_plan(
            archive,
            source_group_path=source_path,
            selection_run=selected.selection_id,
        )
