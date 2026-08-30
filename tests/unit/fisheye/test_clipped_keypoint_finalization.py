from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.shared.zarr.clipped_keypoint_finalization import (
    CLIPPED_KEYPOINT_RESULT_RECEIPT_SCHEMA_ID,
    ClipTerminalKeypointResult,
    ClippedKeypointFinalizationError,
    clip_terminal_result_from_yolo_arrays,
    clipped_keypoint_binding_digests,
    prepare_clipped_keypoint_finalization,
    publish_selector_ineligible_clipped_keypoint_chain,
    validate_clipped_keypoint_finalization_receipt,
    validate_clip_terminal_result_receipt,
)
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_sha256,
)
from fisheye.shared.run_provenance import build_run_provenance
from fisheye.shared.zarr.keypoint_publication_mode import (
    KEYPOINT_PUBLICATION_MODE_PRODUCTION_CANDIDATE,
    KeypointChainPublicationDispositions,
    KeypointPublicationDisposition,
)
from fisheye.shared.zarr.keypoint_bundle_production_publication import (
    publish_keypoint_v2_production_candidate_chain,
)
from fisheye.shared.zarr.crop_shadow import CropGeometryShadowPublication
from fisheye.shared.zarr.crop_storage import plan_crop_geometry_storage
from fisheye.shared.zarr.keypoint_manifest import KeypointPreprocessingReference
from fisheye.shared.zarr.keypoint_successor import TerminalKeypointInferenceBatch
from fisheye.shared.zarr.refined_keypoint_manifest import (
    initial_refined_keypoint_snapshot_identity,
)
from fisheye.shared.zarr_helpers import consolidate_metadata_capture_expected_warnings
from fisheye.shared.zarr_run_completion import (
    COMPLETION_EPOCH_STRICT,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)
from fisheye.utils import write_keypoint_clip_terminal_receipt as receipt_cli
from tests.unit.fisheye.test_keypoint_publication import (
    _crop_fixture,
    _pose_binding,
)


def _preprocessing() -> KeypointPreprocessingReference:
    return KeypointPreprocessingReference(
        profile_id="yolo_pose_crop_v1",
        profile_version=1,
        input_mode="crop_pixel_package",
        document={
            "decoded_dtype": "uint8",
            "channels": "grayscale",
            "resize": "letterbox_model_shape",
            "normalization": "ultralytics_runtime_v1",
        },
    )


def _crop(tmp_path: object) -> CropGeometryShadowPublication:
    dimensions, arrays, manifest = _crop_fixture()
    return CropGeometryShadowPublication(
        output_path=tmp_path / "crop.zarr",  # type: ignore[operator]
        run_id="crop_v2_source",
        dimensions=dimensions,
        plans=plan_crop_geometry_storage(dimensions),
        manifest=manifest,
        arrays=arrays,
        source_manifest={},
        source_arrays={},
        receipt={
            "logical_content_digest": manifest["payload"]["logical_content"][
                "digest"
            ]
        },
    )


def _clip_results(
    crop: CropGeometryShadowPublication,
) -> tuple[ClipTerminalKeypointResult, ...]:
    binding = _pose_binding()
    preprocessing = _preprocessing()
    digests = clipped_keypoint_binding_digests(
        crop=crop,
        pose_model_schema_binding=binding,
        preprocessing=preprocessing,
    )
    points = np.asarray(
        [
            [[5, 10], [15, 5], [15, 15]],
            [[6, 11], [16, 6], [16, 16]],
            [[7, 12], [17, 7], [17, 17]],
            [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]],
        ],
        dtype=np.float32,
    )
    confidence = np.asarray(
        [[0.9, 0.9, 0.9], [0.8, 0.8, 0.8], [0.95, 0.9, 0.85], [np.nan] * 3],
        dtype=np.float32,
    )
    pose_confidence = np.asarray([0.95, 0.85, 0.97, np.nan], dtype=np.float32)
    bbox = np.asarray(
        [[1, 1, 19, 19], [1, 1, 19, 19], [1, 1, 19, 19], [np.nan] * 4],
        dtype=np.float32,
    )
    success = np.asarray([True, True, True, False], dtype=bool)
    keys = np.asarray(crop.arrays["instance_key"])
    signatures = np.asarray(crop.arrays["source_row_signature"])
    results = []
    # Deliberately reverse rows inside each clip. Finalization must recover the
    # canonical recording-level crop order by instance_key.
    for clip_index, row_ids in enumerate(((1, 0), (3, 2))):
        rows = np.asarray(row_ids, dtype=np.int64)
        inference = TerminalKeypointInferenceBatch(
            instance_key=keys[rows].copy(),
            keypoints_roi=points[rows].copy(),
            keypoint_confidences=confidence[rows].copy(),
            pose_confidence=pose_confidence[rows].copy(),
            pose_bbox_xyxy_roi=bbox[rows].copy(),
            pose_success=success[rows].copy(),
        )
        results.append(
            ClipTerminalKeypointResult(
                clip_id=f"clip_{clip_index}",
                clip_index=clip_index,
                terminal_status="complete",
                inference=inference,
                source_crop_row_signature=signatures[rows].copy(),
                crop_run_id=crop.run_id,
                crop_manifest_digest=digests["crop_manifest_digest"],
                crop_coordinate_catalog_digest=digests[
                    "crop_coordinate_catalog_digest"
                ],
                keypoint_coordinate_catalog_digest=digests[
                    "keypoint_coordinate_catalog_digest"
                ],
                pose_model_binding_digest=digests["pose_model_binding_digest"],
                preprocessing_digest=digests["preprocessing_digest"],
                input_package_manifest_digest=(
                    f"{clip_index + 1:064x}"
                ),
            )
        )
    return tuple(results)


def test_clipped_preparation_restores_crop_order_and_keeps_terminal_failures(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crop = _crop(tmp_path)
    monkeypatch.setattr(
        "fisheye.shared.zarr.clipped_keypoint_finalization."
        "validate_crop_geometry_shadow_publication",
        lambda publication: (),
    )

    prepared = prepare_clipped_keypoint_finalization(
        crop,
        _clip_results(crop),
        pose_model_schema_binding=_pose_binding(),
        preprocessing=_preprocessing(),
    )

    arrays = prepared.prepared.arrays
    np.testing.assert_array_equal(arrays["instance_key"], crop.arrays["instance_key"])
    np.testing.assert_array_equal(arrays["source_crop_row_ids"], np.arange(4))
    assert arrays["pose_success"].tolist() == [True, True, True, False]
    assert np.all(np.isnan(arrays["keypoints_roi"][3]))
    assert prepared.preparation_receipt["pending_row_count"] == 0
    assert prepared.preparation_receipt["terminal_failure_count"] == 1


def test_clipped_preparation_rejects_tampered_crop_signature(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crop = _crop(tmp_path)
    monkeypatch.setattr(
        "fisheye.shared.zarr.clipped_keypoint_finalization."
        "validate_crop_geometry_shadow_publication",
        lambda publication: (),
    )
    clips = list(_clip_results(crop))
    changed = np.array(clips[0].source_crop_row_signature, copy=True)
    changed[0, 0] ^= np.uint8(1)
    clips[0] = ClipTerminalKeypointResult(
        **{
            **clips[0].__dict__,
            "source_crop_row_signature": changed,
        }
    )

    with pytest.raises(ClippedKeypointFinalizationError, match="crop signature"):
        prepare_clipped_keypoint_finalization(
            crop,
            clips,
            pose_model_schema_binding=_pose_binding(),
            preprocessing=_preprocessing(),
        )


def test_clipped_preparation_rejects_missing_rows_and_binding_drift(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crop = _crop(tmp_path)
    monkeypatch.setattr(
        "fisheye.shared.zarr.clipped_keypoint_finalization."
        "validate_crop_geometry_shadow_publication",
        lambda publication: (),
    )
    clips = _clip_results(crop)

    with pytest.raises(ClippedKeypointFinalizationError, match="partition"):
        prepare_clipped_keypoint_finalization(
            crop,
            clips[:1],
            pose_model_schema_binding=_pose_binding(),
            preprocessing=_preprocessing(),
        )

    drifted = list(clips)
    drifted[1] = ClipTerminalKeypointResult(
        **{
            **drifted[1].__dict__,
            "preprocessing_digest": "f" * 64,
        }
    )
    with pytest.raises(ClippedKeypointFinalizationError, match="preprocessing"):
        prepare_clipped_keypoint_finalization(
            crop,
            drifted,
            pose_model_schema_binding=_pose_binding(),
            preprocessing=_preprocessing(),
        )


def test_clip_terminal_receipt_rejects_recomputed_source_hash_tampering(
    tmp_path: object,
) -> None:
    crop = _crop(tmp_path)
    result = _clip_results(crop)[0]
    hashes = {
        "instance_key": "1" * 64,
        "source_crop_row_ids": "2" * 64,
        "frame_indices": "3" * 64,
        "keypoints_roi": "4" * 64,
        "keypoint_confidences": "5" * 64,
        "confidence": "6" * 64,
        "pose_bbox_xyxy_roi": "7" * 64,
        "detection_success": "8" * 64,
    }
    crop_hashes = {
        "instance_key": "a" * 64,
        "source_crop_row_ids": "b" * 64,
        "frame_indices": "c" * 64,
        "roi_coordinates_full": "d" * 64,
    }
    payload = {
        "status": "complete",
        "analysis_zarr": "/tmp/example_analysis.zarr",
        "source_group_path": "keypoint_shard_runs/clip_0",
        "source_crop_group_path": "crop_runs/proxy_clip_0",
        "source_coordinate_contract_mode": "legacy_noncanonical",
        "source_array_hashes": hashes,
        "source_crop_array_hashes": crop_hashes,
        "source_crop_roi_shape_hw": [20, 20],
        "input_package_manifest_path": "/tmp/example-package.json",
        "result": result.as_manifest(),
        "production_state_changes": [],
    }
    receipt = {
        "schema_id": CLIPPED_KEYPOINT_RESULT_RECEIPT_SCHEMA_ID,
        "schema_version": 1,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }
    assert validate_clip_terminal_result_receipt(receipt) == ()

    tampered = copy.deepcopy(receipt)
    tampered["payload"]["source_array_hashes"]["instance_key"] = "f" * 64
    assert "clip terminal receipt payload digest mismatch" in (
        validate_clip_terminal_result_receipt(tampered)
    )


def test_legacy_clip_adapter_requires_exact_source_crop_geometry(
    tmp_path: object,
) -> None:
    crop = _crop(tmp_path)
    rows = np.asarray([1, 0], dtype=np.int64)
    source_crop = {
        path: np.asarray(crop.arrays[path])[rows].copy()
        for path in (
            "instance_key",
            "frame_indices",
            "roi_coordinates_full",
            "roi_sizes_full",
        )
    }
    source_crop["source_crop_row_ids"] = np.arange(2, dtype=np.int64)
    points = np.asarray(
        [
            [[5, 10], [15, 5], [15, 15]],
            [[6, 11], [16, 6], [16, 16]],
        ],
        dtype=np.float64,
    )
    yolo = {
        "instance_key": source_crop["instance_key"].copy(),
        "source_crop_row_ids": np.arange(2, dtype=np.int64),
        "frame_indices": source_crop["frame_indices"].copy(),
        "keypoints_roi": points,
        "keypoint_confidences": np.full((2, 3), 0.9, dtype=np.float64),
        "confidence": np.full(2, 0.95, dtype=np.float64),
        "pose_bbox_xyxy_roi": np.asarray(
            [[1, 1, 19, 19], [1, 1, 19, 19]], dtype=np.float32
        ),
        "detection_success": np.ones(2, dtype=bool),
    }

    result = clip_terminal_result_from_yolo_arrays(
        crop,
        yolo,
        source_crop_arrays=source_crop,
        clip_id="clip_0",
        clip_index=0,
        pose_model_schema_binding=_pose_binding(),
        preprocessing=_preprocessing(),
        input_package_manifest_digest="a" * 64,
    )

    assert result.inference.keypoints_roi.dtype == np.dtype(np.float32)
    assert result.inference.instance_key.tolist() == [102, 101]
    changed = {
        **source_crop,
        "roi_coordinates_full": source_crop["roi_coordinates_full"].copy(),
    }
    changed["roi_coordinates_full"][0, 0] += 1
    with pytest.raises(ClippedKeypointFinalizationError, match="origins"):
        clip_terminal_result_from_yolo_arrays(
            crop,
            yolo,
            source_crop_arrays=changed,
            clip_id="clip_0",
            clip_index=0,
            pose_model_schema_binding=_pose_binding(),
            preprocessing=_preprocessing(),
            input_package_manifest_digest="a" * 64,
        )


def test_clip_adapter_accepts_selected_rows_from_direct_strict_crop_v2(
    tmp_path: object,
) -> None:
    crop = _crop(tmp_path)
    rows = np.asarray([1, 3], dtype=np.int64)
    row_count = int(np.asarray(crop.arrays["instance_key"]).shape[0])
    source_crop = {
        "instance_key": np.asarray(crop.arrays["instance_key"]),
        "source_crop_row_ids": np.arange(row_count, dtype=np.int64),
        "frame_indices": np.asarray(crop.arrays["frame_indices"]),
        "roi_coordinates_full": np.asarray(crop.arrays["roi_coordinates_full"]),
        "roi_sizes_full": np.asarray(crop.arrays["roi_sizes_full"]),
    }
    yolo = {
        "instance_key": source_crop["instance_key"][rows].copy(),
        "source_crop_row_ids": rows.copy(),
        "frame_indices": source_crop["frame_indices"][rows].copy(),
        "keypoints_roi": np.zeros((2, 3, 2), dtype=np.float32),
        "keypoint_confidences": np.ones((2, 3), dtype=np.float32),
        "confidence": np.ones(2, dtype=np.float32),
        "pose_bbox_xyxy_roi": np.ones((2, 4), dtype=np.float32),
        "detection_success": np.ones(2, dtype=bool),
    }

    result = clip_terminal_result_from_yolo_arrays(
        crop,
        yolo,
        source_crop_arrays=source_crop,
        clip_id="clip_1",
        clip_index=1,
        pose_model_schema_binding=_pose_binding(),
        preprocessing=_preprocessing(),
        input_package_manifest_digest="c" * 64,
    )

    np.testing.assert_array_equal(
        result.inference.instance_key,
        np.asarray(crop.arrays["instance_key"])[rows],
    )
    np.testing.assert_array_equal(
        result.source_crop_row_signature,
        np.asarray(crop.arrays["source_row_signature"])[rows],
    )


def test_terminal_receipt_supports_direct_strict_crop_without_legacy_row_array(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    crop = _crop(tmp_path)
    rows = np.asarray([1, 3], dtype=np.int64)
    binding = _pose_binding()
    transform = {
        "name": "identity",
        "native_shape_hw": [20, 20],
        "model_shape_hw": [20, 20],
        "pad_top": 0,
        "pad_bottom": 0,
        "pad_left": 0,
        "pad_right": 0,
        "coordinate_mapping": "native_xy = model_xy - [pad_left, pad_top]",
    }
    preprocessing = KeypointPreprocessingReference(
        profile_id="direct_strict_crop_fixture",
        profile_version=1,
        input_mode="crop_pixel_work_package",
        document={
            "clip_source_contract": {
                "coordinate_contract_mode": "legacy_noncanonical",
                "input_mode_effective": "numpy-list",
                "model_input_transform": transform,
            }
        },
    )

    class _Group(dict):
        def __init__(self, *args: object, attrs: dict[str, object], **kwargs: object):
            super().__init__(*args, **kwargs)
            self.attrs = attrs

    package_path = tmp_path / "package.json"
    package_path.write_text("{}\n", encoding="utf-8")
    preprocessing_path = tmp_path / "preprocessing.json"
    preprocessing_path.write_text(
        json.dumps(preprocessing.as_manifest()), encoding="utf-8"
    )
    pose_path = tmp_path / "pose.json"
    pose_path.write_text("{}\n", encoding="utf-8")
    source = _Group(
        {
            "instance_key": np.asarray(crop.arrays["instance_key"])[rows],
            "source_crop_row_ids": rows.copy(),
            "frame_indices": np.asarray(crop.arrays["frame_indices"])[rows],
            "keypoints_roi": np.zeros((2, 3, 2), dtype=np.float32),
            "keypoint_confidences": np.ones((2, 3), dtype=np.float32),
            "confidence": np.ones(2, dtype=np.float32),
            "pose_bbox_xyxy_roi": np.ones((2, 4), dtype=np.float32),
            "detection_success": np.ones(2, dtype=bool),
        },
        attrs={
            "coordinate_contract_mode": "legacy_noncanonical",
            "input_mode_effective": "numpy-list",
            "model_input_transform": transform,
            "source_crop_pixel_work_package_manifest": str(package_path),
            "source_crop_run": crop.run_id,
            "provenance": {
                "model_resolution": {
                    "artifacts": {"model_pose_schema_binding": binding}
                }
            },
        },
    )
    crop_group = _Group(
        {name: np.asarray(value) for name, value in crop.arrays.items()},
        attrs={"run_manifest": crop.manifest},
    )
    root = {
        "keypoint_shard_runs": {"clip_1": source},
        "crop_runs": {crop.run_id: crop_group},
    }
    monkeypatch.setattr(receipt_cli.zarr, "open_group", lambda *_a, **_k: root)
    monkeypatch.setattr(
        receipt_cli,
        "open_persisted_crop_geometry_publication",
        lambda *_a, **_k: crop,
    )
    monkeypatch.setattr(
        receipt_cli, "load_pose_model_schema_binding", lambda _path: binding
    )
    monkeypatch.setattr(receipt_cli, "is_run_complete", lambda _group: True)

    receipt = receipt_cli.build_clip_terminal_receipt(
        analysis_zarr=tmp_path / "analysis.zarr",
        crop_run_id=crop.run_id,
        source_group_path="keypoint_shard_runs/clip_1",
        clip_id="clip_1",
        clip_index=1,
        pose_binding_path=pose_path,
        preprocessing_path=preprocessing_path,
        input_package_manifest_path=package_path,
    )

    assert receipt["payload"]["source_crop_group_path"] == (
        f"crop_runs/{crop.run_id}"
    )
    assert validate_clip_terminal_result_receipt(receipt) == ()


def test_complete_chain_uses_shared_plans_and_one_selector_ineligible_receipt(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crop = _crop(tmp_path)
    monkeypatch.setattr(
        "fisheye.shared.zarr.clipped_keypoint_finalization."
        "validate_crop_geometry_shadow_publication",
        lambda publication: (),
    )
    identity = initial_refined_keypoint_snapshot_identity(
        recording_identity="keypoint_v2_canary",
        lineage_id="33333333-3333-4333-8333-333333333333",
        snapshot_id="44444444-4444-4444-8444-444444444444",
    )

    chain = publish_selector_ineligible_clipped_keypoint_chain(
        crop,
        _clip_results(crop),
        pose_model_schema_binding=_pose_binding(),
        preprocessing=_preprocessing(),
        bundle_root=tmp_path / "finalized",  # type: ignore[operator]
        raw_run_id="raw_v2",
        quality_run_id="quality_v1",
        refined_run_id="refined_v2",
        body_frame_run_id="body_frame_v1",
        refined_identity=identity,
        created_by="pytest",
    )

    assert chain.receipt_path.is_file()
    assert validate_clipped_keypoint_finalization_receipt(chain.receipt) == ()
    assert chain.receipt["payload"]["preparation"]["pending_row_count"] == 0
    assert set(chain.receipt["payload"]["outputs"]) == {
        "raw_keypoints",
        "keypoint_quality",
        "refined_keypoints",
        "body_frame",
    }
    for publication in (chain.raw, chain.quality, chain.refined, chain.body_frame):
        assert publication.plans.profile.profile_id == "published_http_v1"
        assert publication.manifest["payload"]["publication"][
            "stage_selector_eligible"
        ] is False
    assert all(
        item["selector_eligible"] is False
        for item in chain.receipt["payload"]["outputs"].values()
    )

    tampered = copy.deepcopy(chain.receipt)
    tampered["payload"]["outputs"]["raw_keypoints"][
        "manifest_payload_digest"
    ] = "0" * 64
    errors = validate_clipped_keypoint_finalization_receipt(tampered)
    assert "clipped finalization payload digest mismatch" in errors


def test_complete_chain_can_seal_production_candidates_without_selecting_them(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crop = _crop(tmp_path)
    monkeypatch.setattr(
        "fisheye.shared.zarr.clipped_keypoint_finalization."
        "validate_crop_geometry_shadow_publication",
        lambda publication: (),
    )
    identity = initial_refined_keypoint_snapshot_identity(
        recording_identity="keypoint_v2_canary",
        lineage_id="33333333-3333-4333-8333-333333333333",
        snapshot_id="44444444-4444-4444-8444-444444444444",
    )
    provenance = build_run_provenance(
        command="pytest production candidate",
        params={"selector_activation": "deferred"},
        input_run_ids={"crop": crop.run_id},
        cwd=Path.cwd(),
    )

    def disposition(owner: str) -> KeypointPublicationDisposition:
        return KeypointPublicationDisposition(
            mode=KEYPOINT_PUBLICATION_MODE_PRODUCTION_CANDIDATE,
            publication_owner_uuid=owner,
            run_provenance=provenance,
        )

    chain = publish_selector_ineligible_clipped_keypoint_chain(
        crop,
        _clip_results(crop),
        pose_model_schema_binding=_pose_binding(),
        preprocessing=_preprocessing(),
        bundle_root=tmp_path / "production",  # type: ignore[operator]
        raw_run_id="raw_v2",
        quality_run_id="quality_v1",
        refined_run_id="refined_v2",
        body_frame_run_id="body_frame_v1",
        refined_identity=identity,
        created_by="pytest",
        dispositions=KeypointChainPublicationDispositions(
            raw=disposition("a" * 32),
            quality=disposition("b" * 32),
            refined=disposition("c" * 32),
            body_frame=disposition("d" * 32),
        ),
    )

    for publication, parent_path, owner in zip(
        (chain.raw, chain.quality, chain.refined, chain.body_frame),
        (
            "keypoints_runs",
            "keypoint_quality_runs",
            "refined_keypoints_runs",
            "analysis/body_frame_runs",
        ),
        ("a" * 32, "b" * 32, "c" * 32, "d" * 32),
        strict=True,
    ):
        run = zarr.open_group(
            str(publication.output_path / parent_path / publication.run_id),
            mode="r",
            use_consolidated=False,
        )
        assert run.attrs["production_candidate"] is True
        assert run.attrs["stage_selector_eligible"] is False
        assert run.attrs["atomic_publication_owner_uuid"] == owner
        assert run.attrs["palette_run_completion_status"] == "complete"

    archive = tmp_path / "analysis.zarr"  # type: ignore[operator]
    archive_root = zarr.open_group(str(archive), mode="w", zarr_format=3)
    crop_parent = archive_root.create_group("crop_runs")
    crop_parent.attrs["palette_completion_epoch"] = COMPLETION_EPOCH_STRICT
    crop_run = crop_parent.create_group(crop.run_id)
    crop_run.attrs.update(
        {
            "status": "complete",
            "stage_selector_eligible": False,
            RUN_COMPLETION_STATUS_ATTR: RUN_STATUS_COMPLETE,
            "run_manifest": crop.manifest,
        }
    )
    consolidate_metadata_capture_expected_warnings(archive)
    publication = publish_keypoint_v2_production_candidate_chain(
        analysis_zarr=archive,
        chain=chain,
    )

    assert publication["status"] == "complete"
    assert publication["selector_eligible"] is False
    assert publication["registry_updated"] is False
    consolidated = zarr.open_group(
        str(archive), mode="r", zarr_format=3, use_consolidated=True
    )
    for parent_path, run_id in (
        ("keypoints_runs", "raw_v2"),
        ("keypoint_quality_runs", "quality_v1"),
        ("refined_keypoints_runs", "refined_v2"),
        ("analysis/body_frame_runs", "body_frame_v1"),
    ):
        run = consolidated[f"{parent_path}/{run_id}"]
        assert run.attrs["production_candidate"] is True
        assert run.attrs["stage_selector_eligible"] is False
        assert "cluster_output_staging" not in run.attrs
