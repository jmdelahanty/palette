from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.shared.pixel_frame_authority import (
    load_persisted_acquisition_camera_authority,
    stamp_acquisition_camera_frame,
    stamp_acquisition_import_ownership,
    stamp_source_camera_pixel_frame_authority,
)
from fisheye.shared.observation_coordinate_publication import (
    load_persisted_source_camera_position_surface,
)
from fisheye.shared.zarr import crop_snapshot_publication as module
from fisheye.shared.zarr.crop_manifest import (
    CROP_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
    CROP_RUN_MANIFEST_SCHEMA_ID,
)
from fisheye.shared.zarr.crop_snapshot_publication import (
    CROP_SNAPSHOT_PUBLICATION_SCHEMA_ID,
    publish_crop_geometry_production_candidate,
)
from fisheye.shared.zarr.crop_shadow import (
    open_persisted_crop_geometry_publication,
)
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.shared.zarr_run_completion import (
    COMPLETION_EPOCH_ATTR,
    COMPLETION_EPOCH_STRICT,
)
from tests.unit.fisheye.test_crop_shadow import _pixel, _policy, _refined_source


@dataclass
class _BoundPixels:
    pixel_authority: object
    source_video_path: Path
    binding_document_digest: str = "c" * 64
    fail_on_verification: int | None = None
    verification_count: int = 0

    @property
    def source_video_paths(self) -> tuple[Path, ...]:
        return (self.source_video_path,)

    @property
    def source_index_paths(self) -> tuple[Path, ...]:
        return ()

    @property
    def binding_document(self) -> dict[str, str]:
        return {"provider_profile": "published_external_full_frame_video_v1"}

    def assert_verified(self) -> None:
        self.verification_count += 1
        if self.verification_count == self.fail_on_verification:
            raise RuntimeError("pixel authority drift")


def _install_position_authorities(source) -> None:
    root = zarr.open_group(
        str(source.archive_path), mode="a", use_consolidated=False
    )
    root.attrs.update(
        {
            "recording_id": "crop_shadow_multi_subject",
            "camera_serials": ["cam2010095"],
            "source_video_metadata": {
                "schema_id": "palette.source_video_metadata.v2",
                "layout": "single_video",
                "camera_id": "cam2010095",
                "source_path": "/recording/cams/cam2010095.mp4",
                "width": 100,
                "height": 80,
                "total_frames": 4,
                "locator": {
                    "kind": "recording_relative",
                    "relative_path": "cams/cam2010095.mp4",
                },
                "file_fingerprint": {
                    "strategy": "size_mtime_sha256_v1",
                    "value": "d" * 64,
                    "size_bytes": 1234,
                    "mtime_ns": 5678,
                    "relocation_stable": False,
                },
            },
        }
    )
    authority = root.require_group(
        "analysis/acquisition_camera_frames/cam2010095"
    )
    ownership = stamp_acquisition_import_ownership(root, authority)
    stamp_acquisition_camera_frame(
        root,
        authority,
        import_ownership=ownership,
    )
    _, acquisition = load_persisted_acquisition_camera_authority(
        root,
        expected_camera_id="cam2010095",
    )
    point_frame = root.require_group(
        "analysis/coordinate_frames/source_camera/cam2010095/continuous"
    )
    stamp_source_camera_pixel_frame_authority(
        point_frame,
        frame_id="cam2010095_source_camera",
        pixel_convention="continuous",
        acquisition_frame=acquisition,
    )
    bbox_frame = root.require_group(
        "analysis/coordinate_frames/source_camera/cam2010095/pixel_edge_half_open"
    )
    stamp_source_camera_pixel_frame_authority(
        bbox_frame,
        frame_id="cam2010095_source_camera_pixel_edges",
        pixel_convention="pixel_edge_half_open",
        acquisition_frame=acquisition,
    )
    detection = root.require_group("detect_runs").require_group("detect_source")
    attrs = dict(detection.attrs)
    attrs.setdefault(
        "run_manifest",
        {
            "payload_digest": "a" * 64,
            "payload": {"publication": {"stage_selector_eligible": True}},
        },
    )
    attrs.setdefault("stage_selector_eligible", True)
    detection.attrs.put(attrs)


def _wire_authorities(monkeypatch, source, pixels: _BoundPixels) -> list[Path]:
    _install_position_authorities(source)
    calls: list[Path] = []

    def bind_source(path: Path, **_kwargs):
        calls.append(Path(path))
        return source

    monkeypatch.setattr(module, "bind_refined_detection_crop_source", bind_source)
    monkeypatch.setattr(
        module,
        "bind_refined_crop_source_pixel_authority",
        lambda bound_source, *, expected_camera_identity: pixels,
    )
    return calls


def test_signed_hybrid_provider_can_supply_exact_explicit_origins(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = _refined_source(tmp_path)
    root = zarr.open_group(
        str(source.archive_path), mode="a", use_consolidated=False
    )
    provider = root.require_group("crop_runs").create_group("hybrid_provider")
    provider.attrs.update(
        {
            "status": "completed",
            "stage_selector_eligible": False,
            "provider_record_sha256": "a" * 64,
            "source_refined_run_id": source.run_id,
            "source_refined_manifest_digest": source.manifest["payload_digest"],
        }
    )
    source_paths = {
        "instance_key": "instances/instance_key",
        "source_refined_row_ids": "instances/refined_row_ids",
        "frame_indices": "instances/frame_indices",
        "source_acquisition_frame_index": (
            "instances/source_acquisition_frame_index"
        ),
    }
    for provider_path, source_path in source_paths.items():
        provider.create_array(
            provider_path,
            data=np.asarray(source.arrays[source_path][...]),
        )
    origins = np.asarray(
        [[14, 10], [67, 11], [46, 51], [20, 55]],
        dtype=np.int32,
    )
    provider.create_array("roi_coordinates_full", data=origins)
    provider.create_array("roi_sizes_full", data=np.full((4, 2), 8, np.int32))
    signed = {
        "provider_record_sha256": "a" * 64,
        "row_count": 4,
        "source_row_signature_spec_digest": "b" * 64,
        "source_pixel_fingerprint": "c" * 64,
        "source_rowset_fingerprint": "d" * 64,
    }
    monkeypatch.setattr(
        module,
        "validate_hybrid_crop_signed_identity",
        lambda *_args, **_kwargs: signed,
    )

    observed_origins, policy, binding = module._bind_explicit_origin_provider(
        archive=source.archive_path,
        provider_run_id="hybrid_provider",
        source=source,
        base_policy=_policy(),
    )

    np.testing.assert_array_equal(observed_origins, origins)
    assert policy.payload["schema_version"] == 2
    assert policy.placement_authority["run_id"] == "hybrid_provider"
    assert binding["ordered_refined_coverage_exact"] is True

    provider["instance_key"][0] = np.uint64(999)
    with pytest.raises(ValueError, match="differs from the refined rowset"):
        module._bind_explicit_origin_provider(
            archive=source.archive_path,
            provider_run_id="hybrid_provider",
            source=source,
            base_policy=_policy(),
        )


def test_required_candidate_binds_exact_finalized_gated_refined_authority(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = replace(
        _refined_source(tmp_path),
        selection_mode="explicit_selector_ineligible_benchmark",
    )
    writable_root = zarr.open_group(
        str(source.archive_path), mode="a", use_consolidated=False
    )
    writable_run = writable_root["refined_detect_runs"][source.run_id]
    detection = writable_root.require_group("detect_runs").create_group(
        "detect_source"
    )
    detection.attrs.update(
        {
            "run_manifest": {
                "payload_digest": "a" * 64,
                "payload": {
                    "publication": {"stage_selector_eligible": True},
                },
            },
            "stage_selector_eligible": True,
        }
    )
    source = replace(
        source,
        run_group=writable_run,
        instances_group=writable_run["instances"],
    )
    gate_evidence = {
        "requirement": "required",
        "status": "applied",
        "applied": True,
        "gate_run": "gate_001",
        "source_detection_group_path": "detect_runs/detect_source",
        "selection_run": "selection_001",
        "selection_record_sha256": "a" * 64,
        "row_count": source.dimensions.n_source_detections,
    }
    source.run_group.attrs.update(
        {
            "finalized_recording_authority": True,
            "immutable_snapshot": True,
            "registered_detection_gate_requirement": "required",
            "registered_detection_gate": gate_evidence,
        }
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    pixels = _BoundPixels(
        pixel_authority=_pixel(),
        source_video_path=tmp_path / "camera.mp4",
    )
    _wire_authorities(monkeypatch, source, pixels)
    gate_calls = {}

    def gate_validator(*args, **kwargs):
        assert args == (source.archive_path,)
        gate_calls.update(kwargs)
        return {
            "inside": np.ones(
                source.dimensions.n_source_detections,
                dtype=np.bool_,
            ),
            "gate_run": "gate_001",
            "source_detection_group_path": "detect_runs/detect_source",
            "selection_run": "selection_001",
            "selection_record_sha256": "a" * 64,
            "row_count": source.dimensions.n_source_detections,
        }

    result = publish_crop_geometry_production_candidate(
        analysis_zarr=source.archive_path,
        run_id="crop_required",
        policy=_policy(),
        expected_camera_identity="cam2010095",
        scratch_root=scratch,
        source_refined_run_id=source.run_id,
        registered_gate_requirement="required",
        registered_gate_run="gate_001",
        registered_gate_validator=gate_validator,
    )

    assert result["registered_gate_applied"] is True
    assert result["source_refined_run_id"] == source.run_id
    assert gate_calls["require_modern_operational_selection"] is True
    assert gate_calls["allow_selector_ineligible_source"] is False
    assert "require_comparison_bound_selection" not in gate_calls
    root = zarr.open_group(
        str(source.archive_path), mode="r", use_consolidated=False
    )
    assert root["crop_runs"].attrs[COMPLETION_EPOCH_ATTR] == COMPLETION_EPOCH_STRICT
    crop = root["crop_runs"]["crop_required"]
    assert crop.attrs["source_registered_detection_gate"] == gate_evidence


def test_registered_gate_source_mode_uses_exact_raw_detection_authority(
    tmp_path: Path,
) -> None:
    source = _refined_source(tmp_path)
    root = zarr.open_group(
        str(source.archive_path), mode="a", use_consolidated=False
    )
    detection = root.require_group("detect_runs").create_group("detect_source")
    manifest = {
        "payload_digest": "a" * 64,
        "payload": {"publication": {"stage_selector_eligible": True}},
    }
    detection.attrs.update(
        {
            "run_manifest": manifest,
            "stage_selector_eligible": True,
        }
    )
    evidence = {"source_detection_group_path": "detect_runs/detect_source"}

    assert (
        module._registered_gate_source_allows_selector_ineligible(
            archive=source.archive_path,
            source=source,
            gate_evidence=evidence,
        )
        is False
    )

    with pytest.raises(RuntimeError, match="identities differ"):
        module._registered_gate_source_allows_selector_ineligible(
            archive=source.archive_path,
            source=source,
            gate_evidence={
                "source_detection_group_path": "detect_runs/detect_other"
            },
        )

    detection.attrs["run_manifest"] = {
        **manifest,
        "payload_digest": "b" * 64,
    }
    with pytest.raises(RuntimeError, match="manifest identity changed"):
        module._registered_gate_source_allows_selector_ineligible(
            archive=source.archive_path,
            source=source,
            gate_evidence=evidence,
        )
    detection.attrs["run_manifest"] = manifest

    manifest["payload"]["publication"]["stage_selector_eligible"] = False
    detection.attrs.update(
        {
            "run_manifest": manifest,
            "stage_selector_eligible": False,
            "production_candidate": True,
        }
    )
    assert (
        module._registered_gate_source_allows_selector_ineligible(
            archive=source.archive_path,
            source=source,
            gate_evidence=evidence,
        )
        is True
    )

    detection.attrs["production_candidate"] = False
    with pytest.raises(RuntimeError, match="neither an active authority"):
        module._registered_gate_source_allows_selector_ineligible(
            archive=source.archive_path,
            source=source,
            gate_evidence=evidence,
        )


def test_candidate_is_atomically_imported_consolidated_and_unselected(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = replace(
        _refined_source(tmp_path),
        selection_mode="approved_authoritative_refined_v1",
    )
    archive = source.archive_path
    root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    crop_family = root.create_group("crop_runs")
    crop_family.create_group("existing_crop")
    crop_family.attrs.update(
        {
            "latest": "existing_crop",
            "purpose_selectors": {"inspection": "existing_crop"},
        }
    )
    consolidate_metadata_capture_expected_warnings(archive)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    pixels = _BoundPixels(
        pixel_authority=_pixel(),
        source_video_path=tmp_path / "camera.mp4",
    )
    source_calls = _wire_authorities(monkeypatch, source, pixels)

    result = publish_crop_geometry_production_candidate(
        analysis_zarr=archive,
        run_id="crop_candidate_v2",
        policy=_policy(),
        expected_camera_identity="cam2010095",
        scratch_root=scratch,
    )

    assert result["schema_id"] == CROP_SNAPSHOT_PUBLICATION_SCHEMA_ID
    assert result["status"] == "complete"
    assert result["selector_eligible"] is False
    assert result["registry_updated"] is False
    assert result["storage_profile_id"] == "published_http_v1"
    assert result["source_video_path"] == str(tmp_path / "camera.mp4")
    assert result["source_video_paths"] == [str(tmp_path / "camera.mp4")]
    assert result["source_index_paths"] == []
    assert result["source_pixel_provider_profile"] == (
        "published_external_full_frame_video_v1"
    )
    assert result["validation"] == {
        "local_errors": [],
        "published_errors": [],
        "direct_consolidated_metadata_equal": True,
        "source_and_pixel_authorities_reverified": True,
        "root_attributes_unchanged": True,
        "crop_selector_attributes_unchanged": True,
    }
    assert source_calls == [archive, archive, archive]
    assert pixels.verification_count == 3
    assert list(scratch.iterdir()) == []
    writer_receipt = result["node_local_materialization"]["writer_receipt"]
    assert set(writer_receipt) == {
        "publication_seconds",
        "phase_seconds",
        "per_array_write_seconds",
        "writes",
        "logical_hashes",
        "consolidation",
    }
    assert len(writer_receipt["writes"]) == 13
    assert len(writer_receipt["logical_hashes"]) == 13

    direct_root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    consolidated_root = zarr.open_group(str(archive), mode="r", use_consolidated=True)
    for observed_root in (direct_root, consolidated_root):
        family = observed_root["crop_runs"]
        assert dict(family.attrs) == {
            "latest": "existing_crop",
            "purpose_selectors": {"inspection": "existing_crop"},
        }
        run = family["crop_candidate_v2"]
        assert run.attrs["status"] == "complete"
        assert run.attrs["palette_run_completion_contract"] == (
            "palette.zarr_run_completion.v1"
        )
        assert run.attrs["palette_run_completion_status"] == "complete"
        assert run.attrs["palette_run_name"] == "crop_candidate_v2"
        assert run.attrs["stage_selector_eligible"] is False
        assert run.attrs["production_candidate"] is True
        assert run.attrs["immutable_snapshot"] is True
        assert run.attrs["production_selector_activation"] == "deferred"
        assert "shadow_only" not in run.attrs
        assert run.attrs["run_manifest"]["schema_version"] == (
            CROP_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION
        )
        assert (
            run.attrs["run_manifest"]["payload_digest"]
            == (result["run_manifest_digest"])
        )
        assert len(list(run.arrays())) == 13
        np.testing.assert_array_equal(
            run["frame_row_offsets"][:],
            np.asarray([0, 2, 2, 3, 4], dtype=np.int64),
        )
        for _name, array in run.arrays():
            assert "benchmark_only" not in array.attrs
            assert array.attrs["selector_eligible"] is False

    rebound = open_persisted_crop_geometry_publication(
        archive,
        run_id="crop_candidate_v2",
    )
    assert rebound.run_id == "crop_candidate_v2"
    assert rebound.plans.profile.profile_id == "published_http_v1"
    assert rebound.receipt["persisted_archive_path"] == str(archive)
    surface = load_persisted_source_camera_position_surface(
        direct_root,
        "crop_runs/crop_candidate_v2",
    )
    assert surface.coordinates.coordinate_node.path == (
        "crop_runs/crop_candidate_v2/centers_img_xy"
    )
    assert [
        record.record["schema_id"] for record in surface.coordinates.lineage_records
    ] == [CROP_RUN_MANIFEST_SCHEMA_ID]
    from fisheye.analysis.track_kinematics import (
        load_canonical_offline_position_source,
        resolve_detection_from_path,
    )
    from fisheye.shared.observation_coordinate_publication import (
        resolve_source_detection_rowset_from_position_coordinates,
    )

    offline = load_canonical_offline_position_source(
        direct_root,
        direct_root["crop_runs/crop_candidate_v2"],
        crop_run_name="crop_candidate_v2",
    )
    detection_path = resolve_source_detection_rowset_from_position_coordinates(
        offline.position_surface.coordinates
    )
    assert detection_path == "refined_detect_runs/refined_crop_source"
    detection = resolve_detection_from_path(direct_root, detection_path)
    assert detection.run_name == "refined_crop_source"
    assert detection.source_detect_run == "detect_source"

    mutable_root = zarr.open_group(
        str(archive), mode="a", use_consolidated=False
    )
    crop = mutable_root["crop_runs/crop_candidate_v2"]
    crop.attrs["crop_geometry_selection"] = {}
    with pytest.raises(ValueError, match="exactly one detection, crop"):
        load_persisted_source_camera_position_surface(
            mutable_root,
            "crop_runs/crop_candidate_v2",
        )


def test_manifest_position_authority_rejects_center_tampering(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = replace(
        _refined_source(tmp_path),
        selection_mode="approved_authoritative_refined_v1",
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    pixels = _BoundPixels(
        pixel_authority=_pixel(),
        source_video_path=tmp_path / "camera.mp4",
    )
    _wire_authorities(monkeypatch, source, pixels)
    publish_crop_geometry_production_candidate(
        analysis_zarr=source.archive_path,
        run_id="crop_position_tamper",
        policy=_policy(),
        expected_camera_identity="cam2010095",
        scratch_root=scratch,
    )
    writable = zarr.open_group(
        str(source.archive_path), mode="a", use_consolidated=False
    )
    centers = writable["crop_runs/crop_position_tamper/centers_img_xy"]
    changed = np.asarray(centers[0], dtype=np.float32)
    changed[0] += np.float32(1.0)
    centers[0] = changed

    fresh = zarr.open_group(
        str(source.archive_path), mode="r", use_consolidated=False
    )
    with pytest.raises(
        ValueError,
        match="Persisted crop publication is invalid.*center_projection_mismatch",
    ):
        load_persisted_source_camera_position_surface(
            fresh,
            "crop_runs/crop_position_tamper",
        )


def test_authority_drift_fails_before_archive_import(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = replace(
        _refined_source(tmp_path),
        selection_mode="approved_authoritative_refined_v1",
    )
    archive = source.archive_path
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    pixels = _BoundPixels(
        pixel_authority=_pixel(),
        source_video_path=tmp_path / "camera.mp4",
        fail_on_verification=2,
    )
    _wire_authorities(monkeypatch, source, pixels)

    with pytest.raises(RuntimeError, match="pixel authority drift"):
        publish_crop_geometry_production_candidate(
            analysis_zarr=archive,
            run_id="crop_drifted",
            policy=_policy(),
            expected_camera_identity="cam2010095",
            scratch_root=scratch,
        )

    assert not (archive / "crop_runs" / "crop_drifted").exists()


def test_candidate_refuses_existing_immutable_target(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = replace(
        _refined_source(tmp_path),
        selection_mode="approved_authoritative_refined_v1",
    )
    archive = source.archive_path
    root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    root.require_group("crop_runs").create_group("existing")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    pixels = _BoundPixels(
        pixel_authority=_pixel(),
        source_video_path=tmp_path / "camera.mp4",
    )
    calls = _wire_authorities(monkeypatch, source, pixels)

    with pytest.raises(FileExistsError, match="already exists"):
        publish_crop_geometry_production_candidate(
            analysis_zarr=archive,
            run_id="existing",
            policy=_policy(),
            expected_camera_identity="cam2010095",
            scratch_root=scratch,
        )

    assert calls == []
    assert pixels.verification_count == 0


def test_post_import_failure_retains_unselected_failed_tombstone(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = replace(
        _refined_source(tmp_path),
        selection_mode="approved_authoritative_refined_v1",
    )
    archive = source.archive_path
    root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    family = root.create_group("crop_runs")
    family.attrs["latest"] = "existing_crop"
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    pixels = _BoundPixels(
        pixel_authority=_pixel(),
        source_video_path=tmp_path / "camera.mp4",
    )
    _wire_authorities(monkeypatch, source, pixels)
    real_build = module._build_and_persist_manifest
    build_count = 0

    def fail_final_manifest(**kwargs):
        nonlocal build_count
        build_count += 1
        if build_count == 2:
            raise RuntimeError("final manifest failure")
        return real_build(**kwargs)

    monkeypatch.setattr(module, "_build_and_persist_manifest", fail_final_manifest)

    with pytest.raises(RuntimeError, match="final manifest failure"):
        publish_crop_geometry_production_candidate(
            analysis_zarr=archive,
            run_id="failed_candidate",
            policy=_policy(),
            expected_camera_identity="cam2010095",
            scratch_root=scratch,
        )

    observed = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    observed_family = observed["crop_runs"]
    assert observed_family.attrs["latest"] == "existing_crop"
    failed = observed_family["failed_candidate"]
    assert failed.attrs["status"] == "failed"
    assert failed.attrs["palette_run_completion_status"] == "failed"
    assert failed.attrs["stage_selector_eligible"] is False
    assert failed.attrs["production_candidate"] is False
    assert failed.attrs["production_selector_activation"] == (
        "blocked_failed_publication"
    )
    assert "final manifest failure" in failed.attrs["publication_failure"]
