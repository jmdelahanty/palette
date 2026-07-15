from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.shared.detect_reason_codec import write_reason_columns
from fisheye.shared.mask_store import write_component_rle_mask_store_from_dense
from fisheye.utils.validate_refined_subject_mask_contract import (
    backfill_refined_subject_mask_contract,
    validate_refined_subject_mask_contract,
)


LABELS = ["subject_body", "eye_left", "eye_right", "swim_bladder"]


def _review_payload(state: str = "approved") -> dict[str, object]:
    return {
        "state": state,
        "method": "unit_test",
        "intended_use": "training",
    }


def _make_archive(
    zarr_path: Path,
    *,
    omit_component_arrays: bool = False,
    omit_component_provenance: bool = False,
    omit_metrics: bool = False,
    omit_source_crop_row_ids: bool = False,
) -> None:
    root = zarr.open_group(str(zarr_path), mode="w")
    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_test"
    crop = crop_parent.create_group("crop_test")
    crop.create_array("frame_indices", data=np.asarray([10, 11], dtype=np.int32), overwrite=True)
    crop.create_array("detection_indices", data=np.asarray([0, 1], dtype=np.int32), overwrite=True)
    crop.create_array("source_refined_row_ids", data=np.asarray([100, 101], dtype=np.int64), overwrite=True)
    crop.create_array("source_detect_row_index", data=np.asarray([200, 201], dtype=np.int32), overwrite=True)
    crop.create_array("roi_coordinates_full", data=np.asarray([[2, 3], [4, 5]], dtype=np.int32), overwrite=True)
    parent = root.create_group("refined_subject_masks_runs")
    parent.attrs["latest"] = "refined_subject_test"
    run = parent.create_group("refined_subject_test")
    run.attrs.update(
        {
            "source_subject_mask_run": "subject_test",
            "source_crop_run": "crop_test",
            "label_schema_id": "subject_v1_lr",
            "mask_labels": LABELS,
            "output_semantics": "multilabel",
            "refinement_semantics": "canonical_component_masks",
            "method": "unit_test_refinement",
            "created_at_utc": "2026-04-28T00:00:00+00:00",
            "duration_seconds": 1.0,
            "refined_subject_mask_review_status": _review_payload("approved"),
            "component_review_statuses": {label: _review_payload("approved") for label in LABELS},
        }
    )
    masks = np.zeros((2, len(LABELS), 8, 8), dtype=np.uint8)
    masks[:, 0, 1:7, 1:7] = 1
    masks[:, 1, 2:4, 2:4] = 1
    masks[:, 2, 2:4, 5:7] = 1
    masks[:, 3, 4:6, 3:5] = 1
    run.create_array("frame_indices", data=np.asarray([10, 11], dtype=np.int32), overwrite=True)
    run.create_array("detection_indices", data=np.asarray([0, 1], dtype=np.int32), overwrite=True)
    run.create_array("source_refined_row_ids", data=np.asarray([100, 101], dtype=np.int64), overwrite=True)
    run.create_array("source_detect_row_index", data=np.asarray([200, 201], dtype=np.int32), overwrite=True)
    if not omit_source_crop_row_ids:
        run.create_array("source_crop_row_ids", data=np.asarray([0, 1], dtype=np.int64), overwrite=True)
    run.create_array("detection_source", data=np.asarray([0, 0], dtype=np.int8), overwrite=True)
    run.create_array("masks_roi", data=masks, chunks=(1, 1, 8, 8), overwrite=True)
    run.create_array("available_channels", data=np.ones((len(LABELS),), dtype=bool), overwrite=True)
    run.create_array("edit_applied", data=np.zeros((2, len(LABELS)), dtype=bool), overwrite=True)
    if not omit_metrics:
        metrics = run.create_group("metrics")
        metrics.create_array("mask_present", data=masks.any(axis=(2, 3)), overwrite=True)
        metrics.create_array("area_px", data=masks.sum(axis=(2, 3), dtype=np.int64).astype(np.float32), overwrite=True)
        metrics.create_array("centroid_xy", data=np.zeros((2, len(LABELS), 2), dtype=np.float32), overwrite=True)
        metrics.create_array("centroid_valid", data=np.ones((2, len(LABELS)), dtype=bool), overwrite=True)
        metrics.create_array("bbox_xyxy", data=np.zeros((2, len(LABELS), 4), dtype=np.float32), overwrite=True)
        metrics.create_array("bbox_valid", data=np.ones((2, len(LABELS)), dtype=bool), overwrite=True)

    components = run.create_group("components")
    for index, label in enumerate(LABELS):
        component = components.create_group(label)
        if not omit_component_provenance:
            provenance = component.create_group("provenance")
            provenance.attrs.update(
                {
                    "source_stage": "subject_mask_runs",
                    "source_run": "subject_test",
                    "source_method": "unit_test_source",
                    "source_crop_run": "crop_test",
                }
            )
        write_reason_columns(
            component,
            np.asarray(["clean", "clean"], dtype=object),
            2,
            overwrite=True,
        )
        if omit_component_arrays:
            continue
        component.create_array("mask_present", data=masks[:, index].any(axis=(1, 2)), overwrite=True)
        component.create_array(
            "area_px",
            data=masks[:, index].sum(axis=(1, 2), dtype=np.int64).astype(np.float32),
            overwrite=True,
        )
        component.create_array("edit_applied", data=np.zeros((2,), dtype=bool), overwrite=True)


def _replace_dense_masks_with_rle(zarr_path: Path) -> None:
    root = zarr.open_group(str(zarr_path), mode="a")
    run = root["refined_subject_masks_runs/refined_subject_test"]
    masks = np.asarray(run["masks_roi"][:], dtype=np.uint8)
    del run["masks_roi"]
    write_component_rle_mask_store_from_dense(
        run,
        masks,
        component_names=LABELS,
        encode_row_chunk_size=1,
    )


def test_validate_refined_subject_mask_contract_accepts_modern_run(tmp_path: Path) -> None:
    zarr_path = tmp_path / "archive.zarr"
    _make_archive(zarr_path)

    summary = validate_refined_subject_mask_contract(zarr_path)

    assert summary["valid"] is True
    assert summary["run_name"] == "refined_subject_test"
    assert summary["mask_labels"] == LABELS
    assert summary["available_components"] == LABELS
    assert summary["errors"] == []

    explicit = validate_refined_subject_mask_contract(
        zarr_path,
        run_name="refined_subject_masks_runs/refined_subject_test",
    )
    assert explicit["valid"] is True
    assert explicit["run_name"] == "refined_subject_test"


def test_validate_refined_subject_mask_contract_accepts_compact_mask_store(tmp_path: Path) -> None:
    zarr_path = tmp_path / "archive_compact.zarr"
    _make_archive(zarr_path)
    _replace_dense_masks_with_rle(zarr_path)

    summary = validate_refined_subject_mask_contract(zarr_path)

    assert summary["valid"] is True
    assert summary["errors"] == []
    root = zarr.open_group(str(zarr_path), mode="r")
    run = root["refined_subject_masks_runs/refined_subject_test"]
    assert "masks_roi" not in run
    assert "mask_rle" in run


def test_validate_refined_subject_mask_contract_rejects_source_crop_frame_mismatch(tmp_path: Path) -> None:
    zarr_path = tmp_path / "archive.zarr"
    _make_archive(zarr_path)
    root = zarr.open_group(str(zarr_path), mode="a")
    root["refined_subject_masks_runs/refined_subject_test/source_crop_row_ids"][:] = np.asarray([1, 0], dtype=np.int64)

    summary = validate_refined_subject_mask_contract(zarr_path)

    assert summary["valid"] is False
    assert any(issue["code"] == "source_crop_frame_mismatch" for issue in summary["errors"])


def test_backfill_refined_subject_mask_contract_writes_direct_source_crop_row_ids(tmp_path: Path) -> None:
    zarr_path = tmp_path / "archive_missing_crop_rows.zarr"
    _make_archive(zarr_path, omit_source_crop_row_ids=True)

    before = validate_refined_subject_mask_contract(zarr_path)
    assert before["valid"] is False
    missing = [issue for issue in before["errors"] if issue["path"].endswith("/source_crop_row_ids")]
    assert len(missing) == 1
    assert missing[0]["backfillable"] is True

    backfill = backfill_refined_subject_mask_contract(zarr_path)
    after = validate_refined_subject_mask_contract(zarr_path)

    assert "source_crop_row_ids" in backfill["backfilled"]
    assert backfill["source_crop_row_ids_backfill_policy"] == "direct_row_identity_after_matching_crop_row_arrays"
    assert after["valid"] is True
    root = zarr.open_group(str(zarr_path), mode="r")
    run = root["refined_subject_masks_runs/refined_subject_test"]
    np.testing.assert_array_equal(run["source_crop_row_ids"][:], np.asarray([0, 1], dtype=np.int64))


def test_backfill_refined_subject_mask_contract_refuses_mismatched_direct_rows(tmp_path: Path) -> None:
    zarr_path = tmp_path / "archive_mismatched_crop_rows.zarr"
    _make_archive(zarr_path, omit_source_crop_row_ids=True)
    root = zarr.open_group(str(zarr_path), mode="a")
    root["crop_runs/crop_test/detection_indices"][1] = np.int32(99)

    before = validate_refined_subject_mask_contract(zarr_path)
    assert before["valid"] is False
    missing = [issue for issue in before["errors"] if issue["path"].endswith("/source_crop_row_ids")]
    assert len(missing) == 1
    assert missing[0]["backfillable"] is False

    try:
        backfill_refined_subject_mask_contract(zarr_path)
    except ValueError as exc:
        assert "detection_indices" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected direct-row source_crop_row_ids backfill to be refused.")


def test_validate_refined_subject_mask_contract_rejects_corrupt_compact_mask_store(tmp_path: Path) -> None:
    zarr_path = tmp_path / "archive_compact_corrupt.zarr"
    _make_archive(zarr_path)
    _replace_dense_masks_with_rle(zarr_path)
    root = zarr.open_group(str(zarr_path), mode="a")
    counts = root["refined_subject_masks_runs/refined_subject_test/mask_rle/components/00_subject_body/counts"]
    counts[0] = np.uint32(0)

    summary = validate_refined_subject_mask_contract(zarr_path)

    assert summary["valid"] is False
    assert any(issue["code"] == "invalid_mask_store" for issue in summary["errors"])
    assert any("failed dense materialization" in issue["message"] for issue in summary["errors"])


def test_backfill_refined_subject_mask_contract_repairs_derived_component_arrays(tmp_path: Path) -> None:
    zarr_path = tmp_path / "archive.zarr"
    _make_archive(zarr_path, omit_component_arrays=True)

    before = validate_refined_subject_mask_contract(zarr_path)
    assert before["valid"] is False
    assert any(issue["path"].endswith("/mask_present") for issue in before["errors"])

    backfill = backfill_refined_subject_mask_contract(zarr_path)
    after = validate_refined_subject_mask_contract(zarr_path)

    assert backfill["backfill_count"] == 12
    assert after["valid"] is True
    root = zarr.open_group(str(zarr_path), mode="r")
    run = root["refined_subject_masks_runs"]["refined_subject_test"]
    np.testing.assert_array_equal(
        np.asarray(run["components/eye_left/mask_present"][:], dtype=bool),
        np.asarray([True, True], dtype=bool),
    )
    np.testing.assert_allclose(
        np.asarray(run["components/swim_bladder/area_px"][:], dtype=np.float32),
        np.asarray([4.0, 4.0], dtype=np.float32),
    )


def test_backfill_refined_subject_mask_contract_repairs_compact_run_without_dense_masks(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "archive_compact_missing_metrics.zarr"
    _make_archive(zarr_path, omit_component_arrays=True, omit_metrics=True)
    _replace_dense_masks_with_rle(zarr_path)

    before = validate_refined_subject_mask_contract(zarr_path)
    assert before["valid"] is False
    assert any(issue["path"].endswith("/metrics") for issue in before["errors"])

    backfill = backfill_refined_subject_mask_contract(zarr_path)
    after = validate_refined_subject_mask_contract(zarr_path)

    assert "metrics/mask_present" in backfill["backfilled"]
    assert "metrics/area_px" in backfill["backfilled"]
    assert "components/eye_left/mask_present" in backfill["backfilled"]
    assert "components/swim_bladder/area_px" in backfill["backfilled"]
    assert after["valid"] is True

    root = zarr.open_group(str(zarr_path), mode="r")
    run = root["refined_subject_masks_runs"]["refined_subject_test"]
    assert "masks_roi" not in run
    assert "mask_rle" in run
    np.testing.assert_array_equal(
        np.asarray(run["metrics/mask_present"][:], dtype=bool),
        np.asarray(
            [
                [True, True, True, True],
                [True, True, True, True],
            ],
            dtype=bool,
        ),
    )
    np.testing.assert_allclose(
        np.asarray(run["components/subject_body/area_px"][:], dtype=np.float32),
        np.asarray([36.0, 36.0], dtype=np.float32),
    )


def test_backfill_refined_subject_mask_contract_does_not_fake_component_provenance(tmp_path: Path) -> None:
    zarr_path = tmp_path / "archive.zarr"
    _make_archive(zarr_path, omit_component_provenance=True)

    before = validate_refined_subject_mask_contract(zarr_path)
    assert before["valid"] is False
    assert any(issue["code"] == "missing_component_provenance" for issue in before["errors"])

    backfill = backfill_refined_subject_mask_contract(zarr_path)
    after = validate_refined_subject_mask_contract(zarr_path)

    assert backfill["backfill_count"] == 0
    assert after["valid"] is False
    assert any(issue["code"] == "missing_component_provenance" for issue in after["errors"])
