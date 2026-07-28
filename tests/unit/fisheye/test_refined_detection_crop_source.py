from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.shared.instance_keys import mint_detection_instance_keys
from fisheye.shared.zarr.detection_schema import (
    CanonicalDetectionDimensions,
    derive_canonical_detection_geometry,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.refined_detection_crop_source import (
    REFINED_DETECTION_CROP_COORDINATE_STATUS,
    RefinedDetectionCropSourceError,
    bind_refined_detection_crop_source,
)
from fisheye.shared.zarr.refined_detection_manifest import (
    RefinedDetectionSnapshotLineage,
    RefinedDetectionSourceIdentity,
    build_refined_detection_authority_provenance,
)
from fisheye.shared.zarr.refined_detection_snapshot import (
    publish_selector_ineligible_refined_detection_snapshot,
)
from fisheye.shared.zarr.refined_detection_transition import (
    build_accept_all_refined_detection_root,
)
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.shared.row_source_signature import ROW_SOURCE_SIGNATURE_ARRAY
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)
from fisheye.tracking.refined_detection_crop_handoff import (
    plan_refined_detection_crop_handoff,
)


RECORDING_IDENTITY = "refined_crop_multi_subject"
RUN_ID = "refined_crop_source"


def _publication(tmp_path: Path):
    dimensions = CanonicalDetectionDimensions(
        n_frames=4,
        n_instances=4,
        source_width=100,
        source_height=80,
    )
    frames = np.asarray([0, 0, 2, 3], dtype=np.int32)
    boxes = np.asarray(
        [
            [0.20, 0.20, 0.10, 0.10],
            [0.70, 0.20, 0.10, 0.10],
            [0.50, 0.70, 0.20, 0.10],
            [0.25, 0.75, 0.10, 0.15],
        ],
        dtype=np.float32,
    )
    classes = np.asarray([1, 2, 1, 2], dtype=np.int32)
    bbox_img, centers = derive_canonical_detection_geometry(
        boxes,
        source_width=dimensions.source_width,
        source_height=dimensions.source_height,
    )
    canonical = {
        "instances/frame_indices": frames,
        "instances/source_acquisition_frame_index": frames.astype(np.int64),
        "instances/instance_key": mint_detection_instance_keys(
            recording_identity=RECORDING_IDENTITY,
            frame_indices=frames,
            bbox_norm_coords=boxes,
            class_ids=classes,
        ),
        "instances/bbox_norm_coords": boxes,
        "instances/bbox_img_xyxy": bbox_img,
        "instances/centers_img_xy": centers,
        "instances/scores": np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32),
        "instances/class_ids": classes,
        "instances/frame_row_offsets": np.asarray(
            [0, 2, 2, 3, 4],
            dtype=np.int64,
        ),
    }
    transition = build_accept_all_refined_detection_root(
        canonical,
        dimensions=dimensions,
        recording_identity=RECORDING_IDENTITY,
    )
    root = tmp_path / "snapshots"
    publication = publish_selector_ineligible_refined_detection_snapshot(
        dimensions=transition.dimensions,
        arrays=transition.arrays,
        instance_reason_codes=transition.instance_reason_codes,
        source_reason_codes=transition.source_reason_codes,
        destination=root / "source.zarr",
        run_id=RUN_ID,
        lineage=RefinedDetectionSnapshotLineage(
            lineage_id="11111111-1111-4111-8111-111111111111",
            snapshot_id="22222222-2222-4222-8222-222222222222",
            recording_identity=RECORDING_IDENTITY,
            next_refined_row_id=4,
        ),
        source=RefinedDetectionSourceIdentity(
            run_id="detect_source",
            run_manifest_digest="a" * 64,
            logical_content_digest="b" * 64,
        ),
        created_by="test",
        publication_kind="refined_crop_source_test",
        safe_root=root,
    )
    return publication


def _promote_for_test(publication) -> None:
    root = zarr.open_group(
        str(publication.output_path),
        mode="a",
        zarr_format=3,
        use_consolidated=False,
    )
    run = root[f"refined_detect_runs/{RUN_ID}"]
    manifest = copy.deepcopy(dict(run.attrs["run_manifest"]))
    manifest["payload"]["publication"]["stage_selector_eligible"] = True
    manifest["payload_digest"] = canonical_json_sha256(manifest["payload"])
    run.attrs["run_manifest"] = manifest
    run.attrs["stage_selector_eligible"] = True
    parent = root["refined_detect_runs"]
    parent.attrs["authoritative_run"] = RUN_ID
    parent.attrs["authoritative_run_provenance"] = (
        build_refined_detection_authority_provenance(
            run_id=RUN_ID,
            run_manifest_digest=manifest["payload_digest"],
            approved_by="pytest",
            approved_at_utc="2026-07-27T20:00:00+00:00",
            review_method="refined_crop_handoff_test",
            intended_use="analysis",
        )
    )
    consolidate_metadata_capture_expected_warnings(publication.output_path)


def test_selector_ineligible_source_requires_explicit_benchmark_boundary(
    tmp_path: Path,
) -> None:
    publication = _publication(tmp_path)
    with pytest.raises(
        RefinedDetectionCropSourceError,
        match="selector-eligible",
    ):
        bind_refined_detection_crop_source(
            publication.output_path,
            run_id=RUN_ID,
        )

    source = bind_refined_detection_crop_source(
        publication.output_path,
        run_id=RUN_ID,
        allow_selector_ineligible_benchmark=True,
    )
    assert source.selection_mode == "explicit_selector_ineligible_benchmark"
    assert source.instances_path == f"refined_detect_runs/{RUN_ID}/instances"
    assert source.dimensions.n_instances == 4
    assert source.handoff_manifest["crop_publication_authorized"] is False
    assert (
        source.handoff_manifest["coordinate_status"]
        == REFINED_DETECTION_CROP_COORDINATE_STATUS
    )
    np.testing.assert_array_equal(
        source.instances_group["frame_row_offsets"][:],
        [0, 2, 2, 3, 4],
    )


def test_approved_analysis_authority_is_resolved_without_latest_fallback(
    tmp_path: Path,
) -> None:
    publication = _publication(tmp_path)
    _promote_for_test(publication)

    source = bind_refined_detection_crop_source(publication.output_path)

    assert source.run_id == RUN_ID
    assert source.selection_mode == "approved_authoritative_refined_v1"
    assert source.manifest["payload"]["publication"]["stage_selector_eligible"] is True


def test_tampered_refined_geometry_fails_before_crop_planning(tmp_path: Path) -> None:
    publication = _publication(tmp_path)
    root = zarr.open_group(
        str(publication.output_path),
        mode="a",
        zarr_format=3,
        use_consolidated=False,
    )
    boxes = root[f"refined_detect_runs/{RUN_ID}/instances/bbox_norm_coords"]
    changed = np.asarray(boxes[0], dtype=np.float32)
    changed[0] += np.float32(0.1)
    boxes[0] = changed

    with pytest.raises(
        RefinedDetectionCropSourceError,
        match="publication is invalid",
    ):
        bind_refined_detection_crop_source(
            publication.output_path,
            run_id=RUN_ID,
            allow_selector_ineligible_benchmark=True,
        )


def test_benchmark_mode_never_implicitly_selects_a_run(tmp_path: Path) -> None:
    publication = _publication(tmp_path)
    with pytest.raises(
        RefinedDetectionCropSourceError,
        match="require an explicit run_id",
    ):
        bind_refined_detection_crop_source(
            publication.output_path,
            allow_selector_ineligible_benchmark=True,
        )


def _base_crop(snapshot, *, rows: int, stale_row: int | None = None):
    root = zarr.group(store=zarr.storage.MemoryStore(), zarr_format=3)
    group = root.create_group("base")
    group.attrs[RUN_COMPLETION_STATUS_ATTR] = RUN_STATUS_COMPLETE
    group.attrs["crop_storage_mode"] = "materialized"
    group.attrs.update(snapshot.signature_spec.to_attrs())
    keys = np.asarray(snapshot.instance_keys[:rows], dtype=np.uint64)
    signatures = np.asarray(snapshot.signatures[:rows], dtype=np.uint8)
    if stale_row is not None:
        signatures[stale_row] = 0
    group.create_array("instance_key", data=keys)
    group.create_array(ROW_SOURCE_SIGNATURE_ARRAY, data=signatures)
    group.create_array(
        "roi_images",
        data=np.zeros((rows, 8, 8), dtype=np.uint8),
    )
    return group


def test_handoff_plans_only_the_new_instance_for_computation(tmp_path: Path) -> None:
    publication = _publication(tmp_path)
    source = bind_refined_detection_crop_source(
        publication.output_path,
        run_id=RUN_ID,
        allow_selector_ineligible_benchmark=True,
    )
    initial = plan_refined_detection_crop_handoff(
        source,
        source_pixel_fingerprint="a" * 64,
        roi_size=(8, 8),
        signature_batch_rows=2,
    )
    base = _base_crop(initial.source_snapshot, rows=3)

    successor = plan_refined_detection_crop_handoff(
        source,
        source_pixel_fingerprint="a" * 64,
        roi_size=(8, 8),
        base_crop_group=base,
        signature_batch_rows=2,
    )

    assert successor.keyed_plan.summary()["action_counts"] == {
        "copy": 3,
        "compute": 1,
        "preserve_manual": 0,
    }
    np.testing.assert_array_equal(
        successor.source_snapshot.optional_row_arrays["source_refined_row_ids"],
        [0, 1, 2, 3],
    )
    assert successor.receipt["crop_publication_authorized"] is False


def test_handoff_recomputes_only_a_geometry_signature_mismatch(tmp_path: Path) -> None:
    publication = _publication(tmp_path)
    source = bind_refined_detection_crop_source(
        publication.output_path,
        run_id=RUN_ID,
        allow_selector_ineligible_benchmark=True,
    )
    initial = plan_refined_detection_crop_handoff(
        source,
        source_pixel_fingerprint="a" * 64,
        roi_size=(8, 8),
    )
    base = _base_crop(initial.source_snapshot, rows=4, stale_row=1)

    successor = plan_refined_detection_crop_handoff(
        source,
        source_pixel_fingerprint="a" * 64,
        roi_size=(8, 8),
        base_crop_group=base,
    )

    assert successor.keyed_plan.summary()["action_counts"] == {
        "copy": 3,
        "compute": 1,
        "preserve_manual": 0,
    }
