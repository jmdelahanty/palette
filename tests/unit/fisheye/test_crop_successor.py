from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from fisheye.shared.zarr.crop_manifest import (
    CropPixelAuthority,
    CropRefinedSourceIdentity,
    build_crop_row_source_signatures,
)
from fisheye.shared.zarr.crop_schema import derive_crop_placement_geometry
from fisheye.shared.zarr.crop_shadow import (
    PreparedCropGeometrySnapshot,
    prepare_crop_geometry_from_refined_source,
    publish_selector_ineligible_crop_geometry_snapshot,
)
from fisheye.shared.zarr.crop_successor import (
    CROP_GEOMETRY_SUCCESSOR_PLAN_SCHEMA_ID,
    CROP_GEOMETRY_SUCCESSOR_PUBLICATION_RECEIPT_NAME,
    CropGeometrySuccessorError,
    plan_crop_geometry_successor,
    publish_selector_ineligible_crop_geometry_successor,
    validate_crop_geometry_successor_publication_receipt,
)
from fisheye.shared.zarr.detection_schema import (
    derive_canonical_detection_geometry,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.refined_detection_compaction import (
    compact_frozen_refined_detection_delta_generation,
)
from fisheye.shared.zarr.refined_detection_crop_source import (
    bind_refined_detection_crop_source,
)
from tests.unit.fisheye.test_crop_shadow import _pixel, _policy, _refined_source
from tests.unit.fisheye.test_refined_detection_compaction import (
    DELTA_LINEAGE_ID,
    RECORDING_IDENTITY as COMPACTION_RECORDING_IDENTITY,
    SUCCESSOR_SNAPSHOT_ID,
    _base_publication,
    _frozen_delta,
)


def _values(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return np.asarray(value)
    try:
        return np.asarray(value[...])  # type: ignore[index]
    except (IndexError, KeyError, TypeError):
        return np.asarray(value)


def _parent_publication(tmp_path: Path):
    source = _refined_source(tmp_path)
    prepared = prepare_crop_geometry_from_refined_source(
        source,
        policy=_policy(),
        pixel_authority=_pixel(),
    )
    shadow_root = tmp_path / "crop_successors"
    publication = publish_selector_ineligible_crop_geometry_snapshot(
        prepared,
        destination=shadow_root / "parent.zarr",
        run_id="crop_parent",
        shadow_root=shadow_root,
        coordinate_catalog=True,
    )
    return prepared, publication


def _successor_target(
    parent: PreparedCropGeometrySnapshot,
    *,
    parent_manifest_digest: str | None = None,
) -> PreparedCropGeometrySnapshot:
    arrays = {
        path: np.array(_values(value), copy=True, order="C")
        for path, value in parent.arrays.items()
    }

    # Replace one observation in the same frame with a new detection key.
    old_keys = arrays["instance_key"]
    arrays["instance_key"][1] = np.uint64(int(old_keys.max()) + 1)
    arrays["source_refined_row_ids"][1] = np.int64(
        int(arrays["source_refined_row_ids"].max()) + 1
    )

    # Change one surviving observation's detection geometry while retaining its
    # durable instance_key and refined_row_id.
    arrays["bbox_norm_coords"][2] = np.asarray(
        [0.55, 0.70, 0.20, 0.10],
        dtype=np.float32,
    )
    bbox_img, centers = derive_canonical_detection_geometry(
        arrays["bbox_norm_coords"],
        source_width=parent.dimensions.source_width,
        source_height=parent.dimensions.source_height,
    )
    coordinates, source_crop, bbox_roi = derive_crop_placement_geometry(
        centers,
        bbox_img,
        arrays["roi_sizes_full"],
    )
    arrays["bbox_img_xyxy"] = bbox_img
    arrays["centers_img_xy"] = centers
    arrays["roi_coordinates_full"] = coordinates
    arrays["source_crop_xywh"] = source_crop
    arrays["bbox_roi_xyxy"] = bbox_roi

    source_manifest = copy.deepcopy(parent.source_manifest)
    payload = source_manifest["payload"]
    payload["run_id"] = "refined_crop_successor"
    lineage = payload["snapshot_lineage"]
    lineage["snapshot_id"] = "33333333-3333-4333-8333-333333333333"
    lineage["parent_snapshot"] = {
        "run_id": parent.source.run_id,
        "run_manifest_digest": (
            parent.source.run_manifest_digest
            if parent_manifest_digest is None
            else parent_manifest_digest
        ),
    }
    source_manifest["payload_digest"] = canonical_json_sha256(payload)
    source = CropRefinedSourceIdentity(
        run_id="refined_crop_successor",
        run_manifest_digest=source_manifest["payload_digest"],
        logical_content_digest="d" * 64,
        recording_identity=parent.source.recording_identity,
        lineage_id=parent.source.lineage_id,
        snapshot_id=lineage["snapshot_id"],
    )
    arrays["source_row_signature"] = build_crop_row_source_signatures(
        arrays,
        source=source,
        policy=parent.policy,
        pixel_authority=parent.pixel_authority,
    ).signatures

    source_arrays = {
        path: np.array(_values(value), copy=True, order="C")
        for path, value in parent.source_arrays.items()
    }
    source_arrays.update(
        {
            "instances/instance_key": arrays["instance_key"],
            "instances/refined_row_ids": arrays["source_refined_row_ids"],
            "instances/frame_indices": arrays["frame_indices"],
            "instances/source_acquisition_frame_index": arrays[
                "source_acquisition_frame_index"
            ],
            "instances/frame_row_offsets": arrays["frame_row_offsets"],
            "instances/bbox_norm_coords": arrays["bbox_norm_coords"],
            "instances/bbox_img_xyxy": arrays["bbox_img_xyxy"],
            "instances/centers_img_xy": arrays["centers_img_xy"],
        }
    )
    return replace(
        parent,
        source=source,
        arrays=arrays,
        source_manifest=source_manifest,
        source_arrays=source_arrays,
    )


def test_successor_classifies_reused_added_changed_and_retired_rows(
    tmp_path: Path,
) -> None:
    parent_prepared, parent_publication = _parent_publication(tmp_path)
    target = _successor_target(parent_prepared)

    plan = plan_crop_geometry_successor(parent_publication, target)

    old_keys = np.asarray(parent_prepared.arrays["instance_key"], dtype=np.uint64)
    target_keys = np.asarray(target.arrays["instance_key"], dtype=np.uint64)
    np.testing.assert_array_equal(np.sort(plan.reused_instance_keys), np.sort(old_keys[[0, 3]]))
    np.testing.assert_array_equal(plan.added_instance_keys, target_keys[[1]])
    np.testing.assert_array_equal(plan.changed_instance_keys, target_keys[[2]])
    np.testing.assert_array_equal(plan.retired_instance_keys, old_keys[[1]])
    assert plan.receipt["schema_id"] == CROP_GEOMETRY_SUCCESSOR_PLAN_SCHEMA_ID
    assert plan.receipt["keyed_plan"]["action_counts"] == {
        "copy": 2,
        "compute": 2,
        "preserve_manual": 0,
    }
    assert plan.receipt["instance_keys"]["added"]["count"] == 1
    assert plan.receipt["instance_keys"]["changed"]["count"] == 1
    assert plan.receipt["instance_keys"]["retired"]["count"] == 1
    assert plan.receipt["publication_authorized"] is False
    assert plan.receipt["production_state_changes"] == []


def test_successor_rejects_wrong_immediate_refined_parent(tmp_path: Path) -> None:
    parent_prepared, parent_publication = _parent_publication(tmp_path)
    target = _successor_target(
        parent_prepared,
        parent_manifest_digest="f" * 64,
    )

    with pytest.raises(CropGeometrySuccessorError, match="immediate refined snapshot"):
        plan_crop_geometry_successor(parent_publication, target)


def test_successor_rejects_target_rows_that_differ_from_refined_source(
    tmp_path: Path,
) -> None:
    parent_prepared, parent_publication = _parent_publication(tmp_path)
    target = _successor_target(parent_prepared)
    source_arrays = dict(target.source_arrays)
    source_arrays["instances/instance_key"] = np.asarray(
        parent_prepared.arrays["instance_key"],
        dtype=np.uint64,
    )
    target = replace(target, source_arrays=source_arrays)

    with pytest.raises(CropGeometrySuccessorError, match="instance_key"):
        plan_crop_geometry_successor(parent_publication, target)


def test_real_detection_compaction_publishes_complete_crop_successor(
    tmp_path: Path,
) -> None:
    base = _base_publication(tmp_path)
    compaction_root = tmp_path / "compactions"
    compacted = compact_frozen_refined_detection_delta_generation(
        delta_root=_frozen_delta(tmp_path, base),
        delta_lineage_id=DELTA_LINEAGE_ID,
        generation_ordinal=0,
        base_manifest=base.manifest,
        base_arrays=base.arrays,
        destination=compaction_root / "successor.zarr",
        run_id="refined_successor",
        snapshot_id=SUCCESSOR_SNAPSHOT_ID,
        created_by="crop_successor_test",
        safe_root=compaction_root,
    )
    parent_source = bind_refined_detection_crop_source(
        base.output_path,
        run_id="refined_base",
        allow_selector_ineligible_benchmark=True,
    )
    target_source = bind_refined_detection_crop_source(
        compacted.publication.output_path,
        run_id="refined_successor",
        allow_selector_ineligible_benchmark=True,
        parent_manifest=base.manifest,
        parent_arrays=base.arrays,
    )
    pixels = CropPixelAuthority(
        authority_id="test_camera_video#decode=uint8_v1",
        authority_manifest_digest="c" * 64,
        recording_identity=COMPACTION_RECORDING_IDENTITY,
        camera_identity="cam_test",
        n_frames=4,
        source_width=100,
        source_height=80,
    )
    parent_prepared = prepare_crop_geometry_from_refined_source(
        parent_source,
        policy=_policy(),
        pixel_authority=pixels,
    )
    target_prepared = prepare_crop_geometry_from_refined_source(
        target_source,
        policy=_policy(),
        pixel_authority=pixels,
    )
    crop_root = tmp_path / "real_crop_successor"
    parent_crop = publish_selector_ineligible_crop_geometry_snapshot(
        parent_prepared,
        destination=crop_root / "parent.zarr",
        run_id="crop_parent",
        shadow_root=crop_root,
        coordinate_catalog=True,
    )

    successor = publish_selector_ineligible_crop_geometry_successor(
        parent_crop,
        target_prepared,
        destination=crop_root / "successor.zarr",
        run_id="crop_successor",
        shadow_root=crop_root,
        created_by="crop_successor_test",
    )

    assert successor.plan.keyed_plan.summary()["action_counts"] == {
        "copy": 3,
        "compute": 1,
        "preserve_manual": 0,
    }
    assert successor.plan.added_instance_keys.shape == (1,)
    assert successor.plan.changed_instance_keys.shape == (0,)
    assert successor.plan.retired_instance_keys.shape == (0,)
    assert successor.publication.dimensions.n_instances == 4
    assert successor.receipt["payload"]["selector_eligible"] is False
    assert successor.receipt["payload"]["production_state_changes"] == []
    assert validate_crop_geometry_successor_publication_receipt(successor.receipt) == ()
    assert (
        successor.publication.output_path
        / CROP_GEOMETRY_SUCCESSOR_PUBLICATION_RECEIPT_NAME
    ).is_file()

    tampered = copy.deepcopy(successor.receipt)
    tampered["payload"]["selector_eligible"] = True
    errors = validate_crop_geometry_successor_publication_receipt(tampered)
    assert "crop successor receipt payload digest mismatch" in errors
    assert "crop successor receipt must remain selector-ineligible" in errors
