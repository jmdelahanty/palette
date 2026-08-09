from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import zarr

from fisheye.shared.instance_keys import (
    mint_detection_instance_keys,
    mint_manual_curation_instance_keys,
)
from fisheye.shared.zarr.canonical_detection_shadow import (
    canonical_detection_metadata_declaration_maps,
    publish_legacy_canonical_detection_shadow,
    validate_canonical_detection_shadow_publication,
)
from fisheye.shared.zarr.canonical_detection_manifest import (
    CANONICAL_DETECTION_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
    CANONICAL_DETECTION_NATIVE_RUN_MANIFEST_SCHEMA_VERSION,
    build_canonical_detection_run_manifest,
    build_coordinate_canonical_detection_run_manifest,
    build_native_canonical_detection_run_manifest,
    build_native_detection_source_evidence,
    validate_canonical_detection_run_manifest,
    validate_native_detection_source_evidence,
)
from fisheye.shared.zarr.manifest_digest import (
    canonical_json_sha256,
)
from fisheye.shared.zarr.refined_detection_manifest import (
    REFINED_DETECTION_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
    RefinedDetectionSnapshotLineage,
    validate_refined_detection_run_manifest,
)
from fisheye.shared.zarr.refined_detection_schema import (
    REFINED_DETECTION_SCHEMA_V1,
    SOURCE_DECISION_CODE_MAP,
    SOURCE_KIND_CODE_MAP,
)
from fisheye.shared.zarr.refined_detection_shadow import (
    publish_refined_detection_shadow,
    require_safe_refined_detection_shadow_destination,
)
from fisheye.shared.zarr.refined_detection_transition import (
    build_refined_detection_transition,
)
from fisheye.shared.zarr.storage_profiles import PUBLISHED_HTTP_V1


RECORDING_IDENTITY = "shadow_multi_subject_recording"


def _transition():
    source_bbox = np.asarray(
        [[0.25, 0.5, 0.1, 0.2], [0.75, 0.5, 0.1, 0.2]],
        dtype=np.float64,
    )
    manual_bbox = np.asarray([[0.55, 0.5, 0.1, 0.2]], dtype=np.float32)
    source_frames = np.asarray([1, 3], dtype=np.int32)
    source_classes = np.asarray([1, 3], dtype=np.int32)
    source_keys = mint_detection_instance_keys(
        recording_identity=RECORDING_IDENTITY,
        frame_indices=source_frames,
        bbox_norm_coords=source_bbox.astype(np.float32),
        class_ids=source_classes,
    )
    manual_key = mint_manual_curation_instance_keys(
        recording_identity=RECORDING_IDENTITY,
        refined_row_ids=np.asarray([1], dtype=np.int64),
        frame_indices=np.asarray([1], dtype=np.int32),
        bbox_norm_coords=manual_bbox,
        class_ids=np.asarray([2], dtype=np.int32),
    )[0]
    current = {
        "instances": {
            "frame_indices": np.asarray([1, 1], dtype=np.int32),
            "refined_row_ids": np.asarray([0, 1], dtype=np.int64),
            "bbox_norm_coords": np.asarray(
                [source_bbox[0], manual_bbox[0]],
                dtype=np.float64,
            ),
            "source_kind_codes": np.asarray(
                [
                    SOURCE_KIND_CODE_MAP["raw_detect"],
                    SOURCE_KIND_CODE_MAP["manual"],
                ],
                dtype=np.int8,
            ),
            "manual_edit_flags": np.asarray([False, True], dtype=np.bool_),
            "source_detect_row_index": np.asarray([0, -1], dtype=np.int32),
            "confidence_scores": np.asarray([0.9, 0.0], dtype=np.float32),
            "class_ids": np.asarray([1, 2], dtype=np.int32),
            "instance_key": np.asarray(
                [source_keys[0], manual_key],
                dtype=np.uint64,
            ),
            "reason": np.asarray(["clean", "manual_addition"], dtype=object),
        },
        "source_detections": {
            "source_detect_row_index": np.asarray([0, 1], dtype=np.int32),
            "frame_indices": source_frames,
            "bbox_norm_coords": source_bbox,
            "decision_codes": np.asarray(
                [
                    SOURCE_DECISION_CODE_MAP["accepted"],
                    SOURCE_DECISION_CODE_MAP["filtered"],
                ],
                dtype=np.int8,
            ),
            "resolved_refined_row_id": np.asarray([0, -1], dtype=np.int64),
            "confidence_scores": np.asarray([0.9, 0.8], dtype=np.float32),
            "class_ids": source_classes,
            "instance_key": source_keys,
            "reason": np.asarray(["clean", "filtered_blip"], dtype=object),
        },
    }
    return build_refined_detection_transition(
        current,
        n_frames=4,
        source_width=640,
        source_height=480,
        recording_identity=RECORDING_IDENTITY,
    )


def _lineage() -> RefinedDetectionSnapshotLineage:
    return RefinedDetectionSnapshotLineage(
        lineage_id="11111111-1111-4111-8111-111111111111",
        snapshot_id="22222222-2222-4222-8222-222222222222",
        recording_identity=RECORDING_IDENTITY,
        next_refined_row_id=2,
    )


def _legacy_source(source_path: Path) -> Path:
    source = zarr.open_group(str(source_path), mode="w", zarr_format=3)
    source.attrs.update(
        {
            "source_video_width": 640,
            "source_video_height": 480,
        }
    )
    source.create_array(
        "frame_indices",
        data=np.asarray([1, 3], dtype=np.int32),
    )
    source.create_array(
        "bbox_norm_coords",
        data=np.asarray(
            [[0.25, 0.5, 0.1, 0.2], [0.75, 0.5, 0.1, 0.2]],
            dtype=np.float64,
        ),
    )
    source.create_array(
        "scores",
        data=np.asarray([0.9, 0.8], dtype=np.float32),
    )
    source.create_array(
        "class_ids",
        data=np.asarray([1, 3], dtype=np.int32),
    )
    source.create_array(
        "frame_counts",
        data=np.asarray([0, 1, 0, 1], dtype=np.int32),
    )
    return source_path


def _canonical_source(tmp_path: Path):
    source_path = _legacy_source(tmp_path / "legacy_detect.zarr")
    shadow_root = tmp_path / "canonical-shadows"
    return publish_legacy_canonical_detection_shadow(
        source_group_path=source_path,
        recording_identity=RECORDING_IDENTITY,
        source_run_id="legacy_detect_1",
        destination=shadow_root / "canonical.zarr",
        run_id="detect_shadow_1",
        shadow_root=shadow_root,
    )


def _native_source_evidence(canonical) -> dict[str, object]:
    return build_native_detection_source_evidence(
        dimensions=canonical.dimensions,
        recording_identity=RECORDING_IDENTITY,
        producer_id="fisheye.detection.detect_yolo",
        producer_version="0123456789abcdef",
        source_frame_authority={
            "record_ref": "analysis/acquisition_camera_frames/frame_axis@record",
            "record_sha256": "1" * 64,
        },
        source_pixel_authority={
            "record_ref": "raw_video@source_pixel_authority",
            "record_sha256": "2" * 64,
        },
        model_artifact_sha256="3" * 64,
        run_provenance={
            "schema": "palette.run_provenance.v1",
            "git_sha": "4" * 40,
            "config_hash": "5" * 64,
            "params": {"conf_threshold": 0.5},
            "input_run_ids": {},
            "input_artifacts": [{"role": "detect_model", "sha256": "3" * 64}],
            "command": "fisheye.detection.detect_yolo",
            "fisheye_version": None,
        },
    )


def test_native_canonical_manifest_v2_preserves_legacy_v1_boundary(
    tmp_path: Path,
) -> None:
    canonical = _canonical_source(tmp_path)
    native_evidence = _native_source_evidence(canonical)
    direct, consolidated = canonical_detection_metadata_declaration_maps(
        canonical.output_path,
        run_id=canonical.run_id,
        plans=canonical.plans,
    )

    manifest = build_native_canonical_detection_run_manifest(
        run_id="detect_native_1",
        dimensions=canonical.dimensions,
        storage_plan=canonical.plans,
        arrays=canonical.arrays,
        source_evidence=native_evidence,
        direct_metadata_declarations=direct,
        consolidated_metadata_declarations=consolidated,
        selector_eligible=False,
    )

    assert manifest["schema_version"] == (
        CANONICAL_DETECTION_NATIVE_RUN_MANIFEST_SCHEMA_VERSION
    )
    assert validate_native_detection_source_evidence(
        native_evidence,
        dimensions=canonical.dimensions,
    ) == ()
    assert validate_canonical_detection_run_manifest(manifest) == ()
    with pytest.raises(ValueError, match="legacy source evidence"):
        build_canonical_detection_run_manifest(
            run_id="detect_native_wrong_builder",
            dimensions=canonical.dimensions,
            storage_plan=canonical.plans,
            arrays=canonical.arrays,
            source_evidence=native_evidence,
            direct_metadata_declarations=direct,
            consolidated_metadata_declarations=consolidated,
        )
    with pytest.raises(ValueError, match="native source evidence"):
        build_native_canonical_detection_run_manifest(
            run_id="detect_legacy_wrong_builder",
            dimensions=canonical.dimensions,
            storage_plan=canonical.plans,
            arrays=canonical.arrays,
            source_evidence=canonical.manifest["payload"]["source_evidence"],
            direct_metadata_declarations=direct,
            consolidated_metadata_declarations=consolidated,
        )

    tampered = copy.deepcopy(manifest)
    tampered["payload"]["source_evidence"]["run_provenance"]["document"][
        "command"
    ] = "tampered"
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])
    assert "native run provenance digest mismatch" in (
        validate_canonical_detection_run_manifest(tampered)
    )


def test_opt_in_canonical_v3_binds_catalog_and_explicit_source_kind(
    tmp_path: Path,
) -> None:
    canonical = _canonical_source(tmp_path)
    direct, consolidated = canonical_detection_metadata_declaration_maps(
        canonical.output_path,
        run_id=canonical.run_id,
        plans=canonical.plans,
    )
    manifest = build_coordinate_canonical_detection_run_manifest(
        run_id="detect_coordinate_1",
        dimensions=canonical.dimensions,
        storage_plan=canonical.plans,
        arrays=canonical.arrays,
        source_evidence=canonical.manifest["payload"]["source_evidence"],
        source_evidence_kind="legacy_conversion",
        direct_metadata_declarations=direct,
        consolidated_metadata_declarations=consolidated,
        selector_eligible=False,
    )

    assert manifest["schema_version"] == (
        CANONICAL_DETECTION_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION
    )
    assert manifest["payload"]["source_evidence_kind"] == "legacy_conversion"
    assert validate_canonical_detection_run_manifest(manifest) == ()

    tampered = copy.deepcopy(manifest)
    catalog = tampered["payload"]["coordinate_contract"]
    catalog["document"]["bindings"][0]["semantic_role"] = (
        "sampled_spatial_surface"
    )
    catalog["digest"] = canonical_json_sha256(catalog["document"])
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])
    assert "coordinate catalog differs from the frozen stage catalog" in (
        validate_canonical_detection_run_manifest(tampered)
    )

    native = build_coordinate_canonical_detection_run_manifest(
        run_id="detect_coordinate_native_1",
        dimensions=canonical.dimensions,
        storage_plan=canonical.plans,
        arrays=canonical.arrays,
        source_evidence=_native_source_evidence(canonical),
        source_evidence_kind="native_detection",
        direct_metadata_declarations=direct,
        consolidated_metadata_declarations=consolidated,
        selector_eligible=False,
    )
    assert native["payload"]["source_evidence_kind"] == "native_detection"
    assert validate_canonical_detection_run_manifest(native) == ()


def test_canonical_shadow_can_read_an_exact_local_stage_bound_to_shared_evidence(
    tmp_path: Path,
) -> None:
    evidence_path = _legacy_source(tmp_path / "evidence.zarr")
    staged_path = tmp_path / "staged.zarr"
    shutil.copytree(evidence_path, staged_path)
    shadow_root = tmp_path / "canonical-shadows"

    result = publish_legacy_canonical_detection_shadow(
        source_group_path=staged_path,
        source_evidence_group_path=evidence_path,
        recording_identity=RECORDING_IDENTITY,
        source_run_id="legacy_detect_1",
        destination=shadow_root / "canonical.zarr",
        run_id="detect_shadow_1",
        shadow_root=shadow_root,
        coordinate_catalog=True,
    )

    assert result.manifest["schema_version"] == (
        CANONICAL_DETECTION_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION
    )
    assert "coordinate_contract" in result.manifest["payload"]
    assert result.manifest["payload"]["source_evidence"]["source_group_path"] == str(
        evidence_path.resolve()
    )
    assert result.receipt["storage_profile_id"] == (
        "detection_published_access_aware_v1"
    )
    assert validate_canonical_detection_shadow_publication(result) == ()


def test_canonical_shadow_rejects_a_staged_source_that_differs_from_evidence(
    tmp_path: Path,
) -> None:
    evidence_path = _legacy_source(tmp_path / "evidence.zarr")
    staged_path = tmp_path / "staged.zarr"
    shutil.copytree(evidence_path, staged_path)
    staged = zarr.open_group(str(staged_path), mode="r+", use_consolidated=False)
    staged["class_ids"][0] = np.int32(99)
    destination = tmp_path / "canonical-shadows/canonical.zarr"

    with pytest.raises(ValueError, match="Staged canonical source differs"):
        publish_legacy_canonical_detection_shadow(
            source_group_path=staged_path,
            source_evidence_group_path=evidence_path,
            recording_identity=RECORDING_IDENTITY,
            source_run_id="legacy_detect_1",
            destination=destination,
            run_id="detect_shadow_1",
            shadow_root=destination.parent,
        )

    assert not destination.exists()


def test_shadow_publisher_is_standalone_consolidated_and_selector_ineligible(
    tmp_path: Path,
) -> None:
    shadow_root = tmp_path / "palette-refined-detection-shadows"
    destination = shadow_root / "multi_subject.zarr"

    result = publish_refined_detection_shadow(
        _transition(),
        destination=destination,
        run_id="refined_shadow_1",
        lineage=_lineage(),
        canonical_source=_canonical_source(tmp_path),
        shadow_root=shadow_root,
        coordinate_catalog=True,
    )

    assert result.output_path == destination.resolve()
    assert result.receipt["status"] == "complete"
    assert result.receipt["production_state_changes"] == []
    assert result.receipt["storage_profile_id"] == (
        "detection_published_access_aware_v1"
    )
    assert result.manifest["schema_version"] == (
        REFINED_DETECTION_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION
    )
    assert "coordinate_contract" in result.manifest["payload"]
    assert validate_refined_detection_run_manifest(result.manifest) == ()
    receipt = json.loads(
        (destination / "shadow_publication_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt == result.receipt

    direct_run_metadata = json.loads(
        (destination / "refined_detect_runs/refined_shadow_1/zarr.json").read_text(
            encoding="utf-8"
        )
    )
    archive_metadata = json.loads(
        (destination / "zarr.json").read_text(encoding="utf-8")
    )
    consolidated_run_metadata = archive_metadata["consolidated_metadata"]["metadata"][
        "refined_detect_runs/refined_shadow_1"
    ]
    assert direct_run_metadata.get("consolidated_metadata") is None
    assert consolidated_run_metadata["consolidated_metadata"] == {
        "kind": "inline",
        "must_understand": False,
        "metadata": {},
    }

    root = zarr.open_group(
        str(destination),
        mode="r",
        use_consolidated=True,
    )
    family = root["refined_detect_runs"]
    run = family["refined_shadow_1"]
    assert root.attrs["benchmark_only"] is True
    assert root.attrs["selector_eligible"] is False
    assert family.attrs["selection_contract"] == "none_shadow_direct_path_only"
    assert "latest" not in family.attrs
    assert "latest_complete" not in family.attrs
    assert "authoritative_run" not in family.attrs
    assert run.attrs["status"] == "complete"
    assert run.attrs["stage_selector_eligible"] is False
    assert set(run["instances"].array_keys()) == {
        binding.path.split("/", 1)[1]
        for binding in REFINED_DETECTION_SCHEMA_V1.core_bindings
        if binding.path.startswith("instances/")
    }
    assert np.asarray(run["instances/frame_row_offsets"][:]).tolist() == [
        0,
        0,
        2,
        2,
        2,
    ]
    assert np.unique(np.asarray(run["instances/instance_key"][:])).size == 2


def test_pre_promotion_canonical_profile_remains_manifest_compatible(
    tmp_path: Path,
) -> None:
    source_path = _legacy_source(tmp_path / "legacy_detect.zarr")
    shadow_root = tmp_path / "canonical-shadows"
    result = publish_legacy_canonical_detection_shadow(
        source_group_path=source_path,
        recording_identity=RECORDING_IDENTITY,
        source_run_id="legacy_detect_1",
        destination=shadow_root / "canonical.zarr",
        run_id="detect_shadow_legacy_profile",
        shadow_root=shadow_root,
        profile=PUBLISHED_HTTP_V1,
    )

    assert result.plans.profile is PUBLISHED_HTTP_V1
    assert result.manifest["schema_version"] == 1
    assert "coordinate_contract" not in result.manifest["payload"]
    assert validate_canonical_detection_run_manifest(result.manifest) == ()
    assert validate_canonical_detection_shadow_publication(result) == ()


def test_shadow_destination_rejects_unsafe_existing_or_canonical_paths(
    tmp_path: Path,
) -> None:
    shadow_root = tmp_path / "safe"
    shadow_root.mkdir()

    with pytest.raises(ValueError, match="child"):
        require_safe_refined_detection_shadow_destination(
            shadow_root,
            shadow_root=shadow_root,
        )
    with pytest.raises(ValueError, match="suffix"):
        require_safe_refined_detection_shadow_destination(
            shadow_root / "candidate",
            shadow_root=shadow_root,
        )
    existing = shadow_root / "existing.zarr"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        require_safe_refined_detection_shadow_destination(
            existing,
            shadow_root=shadow_root,
        )
    canonical_parent = shadow_root / "recording_analysis.zarr"
    canonical_parent.mkdir()
    with pytest.raises(ValueError, match="recording archive"):
        require_safe_refined_detection_shadow_destination(
            canonical_parent / "shadow.zarr",
            shadow_root=shadow_root,
        )


def test_shadow_root_must_be_explicitly_safe(tmp_path: Path) -> None:
    unsafe_root = Path("/var/lib/palette-shadows")
    with pytest.raises(ValueError, match="Shadow roots"):
        require_safe_refined_detection_shadow_destination(
            unsafe_root / "candidate.zarr",
            shadow_root=unsafe_root,
        )


def test_canonical_source_manifest_rejects_recomputed_nested_storage_tampering(
    tmp_path: Path,
) -> None:
    source = _canonical_source(tmp_path)
    tampered = copy.deepcopy(source.manifest)
    tampered["payload"]["storage_plan"]["arrays"][0]["plan"]["chunk_nbytes"] += 1
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])

    assert any(
        "storage_plan differs" in error
        for error in validate_canonical_detection_run_manifest(tampered)
    )


def test_canonical_source_revalidation_detects_legacy_source_mutation(
    tmp_path: Path,
) -> None:
    source = _canonical_source(tmp_path)
    legacy_path = Path(
        source.manifest["payload"]["source_evidence"]["source_group_path"]
    )
    legacy = zarr.open_group(str(legacy_path), mode="r+", use_consolidated=False)
    legacy["scores"][0] = np.float32(0.7)

    assert "canonical legacy source evidence changed on disk" in (
        validate_canonical_detection_shadow_publication(source)
    )


def test_refined_shadow_rejects_source_audit_drift_before_writing(
    tmp_path: Path,
) -> None:
    canonical = _canonical_source(tmp_path)
    transition = _transition()
    transition.arrays["instances/scores"][0] = np.float32(0.7)
    transition.arrays["source_detections/scores"][0] = np.float32(0.7)
    shadow_root = tmp_path / "refined-shadows"
    destination = shadow_root / "drift.zarr"

    with pytest.raises(ValueError, match="does not match canonical evidence"):
        publish_refined_detection_shadow(
            transition,
            destination=destination,
            run_id="refined_shadow_drift",
            lineage=_lineage(),
            canonical_source=canonical,
            shadow_root=shadow_root,
        )

    assert not destination.exists()
