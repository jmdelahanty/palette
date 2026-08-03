from __future__ import annotations

from copy import deepcopy

import numpy as np
import zarr

from fisheye.analysis.tail_kinematics_schema import (
    TailKinematicsDimensions,
    build_tail_kinematics_array_declarations,
    stamp_tail_kinematics_array_schema,
    tail_kinematics_fill_values,
    validate_tail_kinematics_array_schema,
)
from fisheye.analysis.tail_kinematics_storage import (
    ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR,
    build_tail_kinematics_storage_receipt,
    create_tail_kinematics_arrays_from_receipt,
    persist_tail_kinematics_storage_receipt,
    validate_tail_kinematics_storage_receipt,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.storage_profiles import PUBLISHED_HTTP_V1


def test_tail_kinematics_declarations_close_core_and_revision_bundle() -> None:
    core = build_tail_kinematics_array_declarations(
        include_source_revision_bundle=False,
        byte_planner_adopted=True,
    )
    with_revisions = build_tail_kinematics_array_declarations(
        include_source_revision_bundle=True,
        byte_planner_adopted=True,
    )

    assert len(core) == 21
    assert len(with_revisions) == 23
    assert {item.path for item in with_revisions} - {item.path for item in core} == {
        "source_refined_subject_masks/row_revision",
        "source_refined_subject_masks/row_revision_available",
    }
    dtype_by_path = {
        item.path: item.contract.dtype.numpy_dtype for item in with_revisions
    }
    assert np.dtype(dtype_by_path["instance_key"]) == np.dtype(np.uint64)
    assert np.dtype(dtype_by_path["source_crop_row_ids"]) == np.dtype(np.int64)
    assert np.dtype(dtype_by_path["source_acquisition_frame_index"]) == np.dtype(
        np.int64
    )
    assert np.dtype(dtype_by_path["tail_angle_rad"]) == np.dtype(np.float32)
    fills = tail_kinematics_fill_values(include_source_revision_bundle=True)
    assert np.isnan(fills["tail_angle_rad"])
    assert fills["valid"] is False
    assert fills["failure_reason_bytes"] == 0
    assert fills["source_refined_subject_masks/row_revision_available"] is False


def test_tail_kinematics_byte_plan_scales_rows_by_uncompressed_record_bytes() -> None:
    dimensions = TailKinematicsDimensions(
        n_rows=1_000_000,
        n_tail_samples=10,
        n_components=4,
    )
    receipt = build_tail_kinematics_storage_receipt(
        dimensions,
        profile=PUBLISHED_HTTP_V1,
    )
    by_path = {entry.declaration.path: entry.plan for entry in receipt.entries}

    assert by_path["source_acquisition_frame_index"].chunk_nbytes >= 512 * 1024
    assert by_path["tail_angle_sample_xy"].chunk_nbytes >= 512 * 1024
    assert (
        by_path["source_acquisition_frame_index"].chunk_shape[0]
        > by_path["tail_angle_sample_xy"].chunk_shape[0]
    )
    assert by_path["tail_angle_sample_xy"].chunk_shape[1:] == (10, 2)
    assert by_path["tail_angle_sample_s"].logical_nbytes == 40
    assert by_path["tail_angle_sample_s"].shard_shape is None
    assert all(
        entry.plan.codec_profile_id == "zstd_fast_v1" for entry in receipt.entries
    )


def test_tail_kinematics_candidate_receipt_replans_and_validates_physical_metadata() -> (
    None
):
    dimensions = TailKinematicsDimensions(
        n_rows=20_000,
        n_tail_samples=10,
        n_components=3,
    )
    receipt = build_tail_kinematics_storage_receipt(
        dimensions,
        profile=PUBLISHED_HTTP_V1,
    )
    root = zarr.group()
    run = root.require_group("analysis/tail_kinematics_runs/candidate")
    create_tail_kinematics_arrays_from_receipt(
        run,
        receipt=receipt,
        dimensions=dimensions,
    )
    stamp_tail_kinematics_array_schema(
        run,
        dimensions,
        byte_planner_adopted=True,
    )
    persist_tail_kinematics_storage_receipt(run, receipt)

    assert (
        validate_tail_kinematics_array_schema(
            run,
            byte_planner_adopted=True,
        )
        == ()
    )
    assert validate_tail_kinematics_storage_receipt(run) == ()
    assert run["tail_angle_rad"].fill_value != run["tail_angle_rad"].fill_value
    assert bool(run["valid"].fill_value) is False

    tampered = deepcopy(dict(run.attrs[ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR]))
    tampered["payload"]["arrays"][0]["plan"]["chunk_shape"][0] += 1
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])
    run.attrs[ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR] = tampered
    errors = validate_tail_kinematics_storage_receipt(run)
    assert errors
    assert any(
        "invalid" in message or "executable planning" in message for message in errors
    )


def test_tail_kinematics_schema_rejects_partial_revision_bundle() -> None:
    dimensions = TailKinematicsDimensions(
        n_rows=3,
        n_tail_samples=10,
        n_components=2,
    )
    receipt = build_tail_kinematics_storage_receipt(
        dimensions,
        profile=PUBLISHED_HTTP_V1,
    )
    root = zarr.group()
    run = root.require_group("analysis/tail_kinematics_runs/candidate")
    create_tail_kinematics_arrays_from_receipt(
        run,
        receipt=receipt,
        dimensions=dimensions,
    )
    del run["source_refined_subject_masks/row_revision_available"]

    errors = validate_tail_kinematics_array_schema(
        run,
        byte_planner_adopted=True,
    )
    assert errors == ("Tail-kinematics source-revision optional bundle is partial.",)
