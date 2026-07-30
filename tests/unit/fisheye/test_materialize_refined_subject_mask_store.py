from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.shared.mask_store import (
    mark_mask_rle_stale_attrs,
    open_mask_store,
    refresh_bitpacked_mask_store_from_dense,
    write_bitpacked_mask_store_from_dense,
    validate_component_rle_mask_store_against_dense,
    validate_component_rle_mask_store_invariants,
    write_component_rle_mask_store_from_dense,
)
from fisheye.utils.materialize_refined_subject_mask_store import (
    materialize_refined_subject_mask_store,
)


def _build_compact_refined_zarr(store_path: Path, *, keep_dense: bool = False) -> tuple[np.ndarray, zarr.Group]:
    root = zarr.open_group(str(store_path), mode="w")
    parent = root.create_group("refined_subject_masks_runs")
    parent.attrs["latest"] = "refined_001"
    run = parent.create_group("refined_001")
    labels = ["subject_body", "eye_left", "eye_right", "swim_bladder"]
    run.attrs["mask_labels"] = labels
    run.create_array("available_channels", data=np.ones((len(labels),), dtype=bool), overwrite=True)
    masks = np.zeros((3, len(labels), 12, 10), dtype=np.uint8)
    masks[0, 0, 1:5, 2:6] = 1
    masks[1, 1, 3:7, 4:8] = 1
    masks[2, 3, 6:10, 1:4] = 1
    dense = run.create_array("masks_roi", data=masks, chunks=(1, 1, 12, 10), overwrite=True)
    write_component_rle_mask_store_from_dense(
        run,
        dense,
        component_names=tuple(labels),
        encode_row_chunk_size=1,
    )
    if not keep_dense:
        del run["masks_roi"]
        run.attrs["mask_store_encodings"] = ["component_rle_v1"]
        run.attrs["mask_storage_encoding"] = "component_rle_v1"
        run.attrs["masks_roi_materialized"] = False
    return masks, root


def test_materialize_refined_subject_mask_store_dry_run_reports_would_materialize(tmp_path: Path) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    _build_compact_refined_zarr(zarr_path)

    summary = materialize_refined_subject_mask_store(zarr_path, apply=False)

    assert summary["status"] == "would_materialize"
    assert summary["has_dense_before"] is False
    reopened = zarr.open_group(str(zarr_path), mode="r")
    assert "masks_roi" not in reopened["refined_subject_masks_runs/refined_001"]


def test_materialize_refined_subject_mask_store_recreates_dense_cache(tmp_path: Path) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    expected, _root = _build_compact_refined_zarr(zarr_path)

    summary = materialize_refined_subject_mask_store(zarr_path, apply=True, chunk_size=2)

    assert summary["status"] == "materialized"
    assert summary["rows_written"] == 3
    reopened = zarr.open_group(str(zarr_path), mode="r")
    run = reopened["refined_subject_masks_runs/refined_001"]
    np.testing.assert_array_equal(np.asarray(run["masks_roi"][:], dtype=np.uint8), expected)
    assert run.attrs["masks_roi_materialized"] is True
    assert run.attrs["mask_store_encodings"] == ["dense_uint8", "component_rle_v1"]
    assert run.attrs["mask_storage_encoding"] == "dense_uint8+component_rle_v1"
    assert run.attrs["masks_roi_materialized_from"] == "mask_rle"
    assert run.attrs["masks_roi_materialization_derivatives_validated"] is False
    assert (
        run.attrs["masks_roi_materialization_freshness_policy"]
        == "preserve_existing_stale_flags_v1"
    )


def test_materialize_dense_authority_preserves_derived_stale_flags(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    _expected, root = _build_compact_refined_zarr(zarr_path)
    run = root["refined_subject_masks_runs/refined_001"]
    run.attrs["derived_mask_caches_stale"] = True
    run.attrs["metrics_stale"] = True
    run.attrs["contours_stale"] = True
    run.attrs["mask_rle_stale"] = False
    run.attrs["derived_mask_caches_stale_reason"] = "preexisting_edit"

    summary = materialize_refined_subject_mask_store(
        zarr_path,
        apply=True,
        chunk_size=2,
    )

    assert summary["status"] == "materialized"
    assert summary["derived_freshness_policy"] == "preserved_not_validated_v1"
    reopened = zarr.open_group(str(zarr_path), mode="r")
    materialized = reopened["refined_subject_masks_runs/refined_001"]
    assert materialized.attrs["derived_mask_caches_stale"] is True
    assert materialized.attrs["metrics_stale"] is True
    assert materialized.attrs["contours_stale"] is True
    assert materialized.attrs["mask_rle_stale"] is False
    assert materialized.attrs["derived_mask_caches_stale_reason"] == "preexisting_edit"


def test_materialize_refined_subject_mask_store_recreates_dense_cache_from_bitpacked(tmp_path: Path) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    expected, root = _build_compact_refined_zarr(zarr_path, keep_dense=True)
    run = root["refined_subject_masks_runs/refined_001"]
    write_bitpacked_mask_store_from_dense(
        run,
        run["masks_roi"],
        component_names=tuple(str(value) for value in run.attrs["mask_labels"]),
        encode_row_chunk_size=2,
        validation_mode="invariants",
    )
    del run["mask_rle"]
    del run["masks_roi"]
    run.attrs["mask_store_encodings"] = ["bitpacked_binary_v1"]
    run.attrs["mask_storage_encoding"] = "bitpacked_binary_v1"
    run.attrs["masks_roi_materialized"] = False
    run.attrs["mask_rle_materialized"] = False

    summary = materialize_refined_subject_mask_store(zarr_path, apply=True, chunk_size=2)

    assert summary["status"] == "materialized"
    reopened = zarr.open_group(str(zarr_path), mode="r")
    run = reopened["refined_subject_masks_runs/refined_001"]
    np.testing.assert_array_equal(np.asarray(run["masks_roi"][:], dtype=np.uint8), expected)
    assert run.attrs["mask_store_encodings"] == ["dense_uint8", "bitpacked_binary_v1"]
    assert run.attrs["mask_storage_encoding"] == "dense_uint8+bitpacked_binary_v1"
    assert run.attrs["masks_roi_materialized_from"] == "mask_bitpacked"


def test_refresh_bitpacked_mask_store_updates_only_selected_rows_and_components(tmp_path: Path) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    expected, root = _build_compact_refined_zarr(zarr_path, keep_dense=True)
    run = root["refined_subject_masks_runs/refined_001"]
    labels = tuple(str(value) for value in run.attrs["mask_labels"])
    write_bitpacked_mask_store_from_dense(
        run,
        run["masks_roi"],
        component_names=labels,
        encode_row_chunk_size=2,
        validation_mode="invariants",
    )
    bitpacked_before = open_mask_store(run, prefer="bitpacked").read_dense()
    updated = expected.copy()
    updated[1, 1] = 0
    updated[1, 1, 2:10, 2:8] = 1
    run["masks_roi"][1, 1] = updated[1, 1]

    summary = refresh_bitpacked_mask_store_from_dense(
        run,
        component_names=labels,
        refresh_components=("eye_left",),
        refresh_rows=(1,),
        source_path="refined_subject_masks_runs/refined_001",
        validation_mode="invariants",
    )

    assert summary["status"] == "bitpacked_refreshed"
    assert summary["refresh_scope"] == "selection"
    assert summary["refreshed_component_names"] == ["eye_left"]
    assert summary["refreshed_rows"] == [1]
    assert summary["mask_bitpacked_validation"]["status"] == "passed"
    bitpacked_after = open_mask_store(run, prefer="bitpacked").read_dense()
    np.testing.assert_array_equal(bitpacked_after[1, 1], updated[1, 1])
    np.testing.assert_array_equal(bitpacked_after[0], bitpacked_before[0])
    np.testing.assert_array_equal(bitpacked_after[2], bitpacked_before[2])
    np.testing.assert_array_equal(bitpacked_after[1, 0], bitpacked_before[1, 0])
    np.testing.assert_array_equal(bitpacked_after[1, 2:], bitpacked_before[1, 2:])


def test_materialize_refined_subject_mask_store_refreshes_scoped_bitpacked_from_dense(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    expected, root = _build_compact_refined_zarr(zarr_path, keep_dense=True)
    run = root["refined_subject_masks_runs/refined_001"]
    labels = tuple(str(value) for value in run.attrs["mask_labels"])
    write_bitpacked_mask_store_from_dense(
        run,
        run["masks_roi"],
        component_names=labels,
        encode_row_chunk_size=2,
        validation_mode="invariants",
    )
    bitpacked_before = open_mask_store(run, prefer="bitpacked").read_dense()
    updated = expected.copy()
    updated[2, 3] = 0
    updated[2, 3, 1:11, 4:9] = 1
    run["masks_roi"][2, 3] = updated[2, 3]

    summary = materialize_refined_subject_mask_store(
        zarr_path,
        apply=True,
        refresh_bitpacked=True,
        components=("swim_bladder",),
        rows=(2,),
        chunk_size=2,
    )

    assert summary["status"] == "bitpacked_refreshed"
    assert summary["refresh_scope"] == "selection"
    assert summary["refreshed_component_names"] == ["swim_bladder"]
    assert summary["refreshed_rows"] == [2]
    reopened = zarr.open_group(str(zarr_path), mode="r")
    refreshed = reopened["refined_subject_masks_runs/refined_001"]
    bitpacked_after = open_mask_store(refreshed, prefer="bitpacked").read_dense()
    np.testing.assert_array_equal(bitpacked_after[2, 3], updated[2, 3])
    np.testing.assert_array_equal(bitpacked_after[0], bitpacked_before[0])
    np.testing.assert_array_equal(bitpacked_after[1], bitpacked_before[1])
    np.testing.assert_array_equal(bitpacked_after[2, :3], bitpacked_before[2, :3])


def test_compact_bitpacked_only_store_can_materialize_edit_and_refresh_bitpacked(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    expected, root = _build_compact_refined_zarr(zarr_path, keep_dense=True)
    run = root["refined_subject_masks_runs/refined_001"]
    labels = tuple(str(value) for value in run.attrs["mask_labels"])
    write_bitpacked_mask_store_from_dense(
        run,
        run["masks_roi"],
        component_names=labels,
        encode_row_chunk_size=2,
        validation_mode="invariants",
    )
    del run["mask_rle"]
    del run["masks_roi"]
    run.attrs["mask_store_encodings"] = ["bitpacked_binary_v1"]
    run.attrs["mask_storage_encoding"] = "bitpacked_binary_v1"
    run.attrs["masks_roi_materialized"] = False

    materialized = materialize_refined_subject_mask_store(zarr_path, apply=True, chunk_size=2)
    reopened = zarr.open_group(str(zarr_path), mode="a")
    refreshed = reopened["refined_subject_masks_runs/refined_001"]
    edited = expected.copy()
    edited[0, 0] = 0
    edited[0, 0, 0:4, 0:4] = 1
    refreshed["masks_roi"][0, 0] = edited[0, 0]
    refresh = materialize_refined_subject_mask_store(
        zarr_path,
        apply=True,
        refresh_bitpacked=True,
        components=("subject_body",),
        rows=(0,),
        chunk_size=2,
    )

    assert materialized["status"] == "materialized"
    assert materialized["source_encoding"] == "bitpacked_binary_v1"
    assert refresh["status"] == "bitpacked_refreshed"
    final = zarr.open_group(str(zarr_path), mode="r")["refined_subject_masks_runs/refined_001"]
    bitpacked_after = open_mask_store(final, prefer="bitpacked").read_dense()
    np.testing.assert_array_equal(bitpacked_after[0, 0], edited[0, 0])
    np.testing.assert_array_equal(bitpacked_after[1:], expected[1:])
    np.testing.assert_array_equal(bitpacked_after[0, 1:], expected[0, 1:])


def test_component_rle_roundtrip_validation_rejects_mismatched_dense_source(tmp_path: Path) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    expected, root = _build_compact_refined_zarr(zarr_path, keep_dense=True)
    run = root["refined_subject_masks_runs/refined_001"]
    altered = expected.copy()
    altered[0, 0, 0, 0] = 1

    with pytest.raises(ValueError, match="RLE round-trip validation failed"):
        validate_component_rle_mask_store_against_dense(
            run,
            altered,
            row_chunk_size=1,
            source_path="refined_subject_masks_runs/refined_001",
        )


def test_component_rle_writer_stamps_storage_attrs_and_clears_stale_marker(tmp_path: Path) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    _expected, root = _build_compact_refined_zarr(zarr_path, keep_dense=True)
    run = root["refined_subject_masks_runs/refined_001"]
    run.attrs["mask_rle_stale"] = True
    run.attrs["mask_rle_stale_reason"] = "old_edit"

    summary = write_component_rle_mask_store_from_dense(
        run,
        run["masks_roi"],
        component_names=tuple(str(value) for value in run.attrs["mask_labels"]),
        encode_row_chunk_size=1,
    )

    assert summary["roundtrip_validation"]["status"] == "passed"
    assert run.attrs["mask_store_encodings"] == ["dense_uint8", "component_rle_v1"]
    assert run.attrs["mask_storage_encoding"] == "dense_uint8+component_rle_v1"
    assert run.attrs["masks_roi_materialized"] is True
    assert run.attrs["mask_rle_materialized"] is True
    assert run.attrs["mask_rle_stale"] is False
    assert "mask_rle_stale_reason" not in run.attrs


def test_component_rle_writer_supports_invariant_validation_mode(tmp_path: Path) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    _expected, root = _build_compact_refined_zarr(zarr_path, keep_dense=True)
    run = root["refined_subject_masks_runs/refined_001"]
    progress_events: list[str] = []

    summary = write_component_rle_mask_store_from_dense(
        run,
        run["masks_roi"],
        component_names=tuple(str(value) for value in run.attrs["mask_labels"]),
        encode_row_chunk_size=1,
        validation_mode="invariants",
        progress_callback=lambda event, **_payload: progress_events.append(event),
    )

    assert summary["mask_rle_validation_mode"] == "invariants"
    assert summary["mask_rle_validation"]["status"] == "passed"
    assert summary["mask_rle_validation"]["logical_dense_validation"] is False
    assert summary["roundtrip_validation"] == {
        "status": "skipped",
        "reason": "validation_mode=invariants",
    }
    assert summary["phase_seconds"]["invariant_validation"] >= 0.0
    assert progress_events[-2:] == [
        "component_rle_invariant_validation_start",
        "component_rle_invariant_validation_end",
    ]


def test_component_rle_invariant_validation_rejects_corrupt_indptr(tmp_path: Path) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    _expected, root = _build_compact_refined_zarr(zarr_path, keep_dense=True)
    run = root["refined_subject_masks_runs/refined_001"]
    component = run["mask_rle/components/00_subject_body"]
    bad_indptr = np.asarray(component["indptr"][:], dtype=np.int64)
    bad_indptr[-1] += 1
    component["indptr"][:] = bad_indptr

    with pytest.raises(ValueError, match="indptr\\[-1\\]"):
        validate_component_rle_mask_store_invariants(
            run,
            expected_shape=run["masks_roi"].shape,
            component_names=tuple(str(value) for value in run.attrs["mask_labels"]),
            source_path="refined_subject_masks_runs/refined_001",
        )


def test_component_rle_writer_process_shards_encode_before_single_parent_write(tmp_path: Path) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    parent = root.create_group("refined_subject_masks_runs")
    run = parent.create_group("refined_001")
    labels = ["subject_body", "eye_left", "eye_right", "swim_bladder"]
    run.attrs["mask_labels"] = labels
    masks = np.zeros((6, len(labels), 12, 10), dtype=np.uint8)
    for row in range(masks.shape[0]):
        masks[row, 0, 1 + row % 3 : 5 + row % 3, 2:6] = 1
        masks[row, 1, 3:7, 1 + row % 4 : 4 + row % 4] = 1
        masks[row, 3, 7:10, row % 5 : row % 5 + 3] = 1
    dense = run.create_array("masks_roi", data=masks, chunks=(2, 1, 12, 10), overwrite=True)
    progress_events: list[str] = []

    summary = write_component_rle_mask_store_from_dense(
        run,
        dense,
        component_names=tuple(labels),
        encode_row_chunk_size=2,
        encode_workers=2,
        source_zarr_path=zarr_path,
        source_run_path="refined_subject_masks_runs/refined_001",
        progress_callback=lambda event, **_payload: progress_events.append(event),
    )

    assert summary["encode_backend"] == "process_shards"
    assert summary["encode_workers"] == 2
    assert summary["encode_shard_count"] == 2
    assert summary["phase_seconds"]["encode"] >= 0.0
    assert summary["phase_seconds"]["parent_write"] >= 0.0
    assert summary["phase_seconds"]["roundtrip_validation"] >= 0.0
    assert progress_events == [
        "component_rle_encode_start",
        "component_rle_encode_end",
        "component_rle_parent_write_start",
        "component_rle_parent_write_end",
        "component_rle_roundtrip_validation_start",
        "component_rle_roundtrip_validation_end",
    ]
    assert summary["roundtrip_validation"]["status"] == "passed"
    reopened = zarr.open_group(str(zarr_path), mode="r")
    store = open_mask_store(reopened["refined_subject_masks_runs/refined_001"], prefer="rle")
    np.testing.assert_array_equal(store.read_dense(), masks)


def test_mark_mask_rle_stale_attrs_records_dense_edit_scope(tmp_path: Path) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    _expected, root = _build_compact_refined_zarr(zarr_path, keep_dense=True)
    run = root["refined_subject_masks_runs/refined_001"]

    marked = mark_mask_rle_stale_attrs(
        run,
        reason="unit_test_dense_edit",
        updated_components=("subject_body", "swim_bladder"),
        updated_rows=(2, 0, 2),
        updated_at_utc="2026-06-19T00:00:00+00:00",
    )

    assert marked is True
    assert run.attrs["mask_rle_stale"] is True
    assert run.attrs["mask_rle_stale_at_utc"] == "2026-06-19T00:00:00+00:00"
    assert run.attrs["mask_rle_stale_reason"] == "unit_test_dense_edit"
    assert run.attrs["mask_rle_stale_component_names"] == ["subject_body", "swim_bladder"]
    assert run.attrs["mask_rle_stale_row_count"] == 3
    assert run.attrs["mask_rle_stale_row_min"] == 0
    assert run.attrs["mask_rle_stale_row_max"] == 2


def test_materialize_refined_subject_mask_store_refuses_stale_compact_only_source(tmp_path: Path) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    _expected, root = _build_compact_refined_zarr(zarr_path)
    run = root["refined_subject_masks_runs/refined_001"]
    run.attrs["mask_rle_stale"] = True

    with pytest.raises(ValueError, match="mask_rle is marked stale"):
        materialize_refined_subject_mask_store(zarr_path, apply=True)

    reopened = zarr.open_group(str(zarr_path), mode="r")
    assert "masks_roi" not in reopened["refined_subject_masks_runs/refined_001"]


def test_materialize_refined_subject_mask_store_refreshes_existing_dense_cache(tmp_path: Path) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    expected, root = _build_compact_refined_zarr(zarr_path, keep_dense=True)
    run = root["refined_subject_masks_runs/refined_001"]
    run["masks_roi"][:] = 0

    existing = materialize_refined_subject_mask_store(zarr_path, apply=True, overwrite=False)
    refreshed = materialize_refined_subject_mask_store(zarr_path, apply=True, overwrite=True, chunk_size=1)

    assert existing["status"] == "existing"
    assert refreshed["status"] == "materialized"
    reopened = zarr.open_group(str(zarr_path), mode="r")
    np.testing.assert_array_equal(
        np.asarray(reopened["refined_subject_masks_runs/refined_001/masks_roi"][:], dtype=np.uint8),
        expected,
    )


def test_materialize_refined_subject_mask_store_refreshes_rle_from_dense_and_clears_stale(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    expected, root = _build_compact_refined_zarr(zarr_path, keep_dense=True)
    run = root["refined_subject_masks_runs/refined_001"]
    updated = expected.copy()
    updated[0, 0] = 0
    updated[0, 3, 1:4, 1:4] = 1
    run["masks_roi"][:] = updated
    run.attrs["mask_rle_stale"] = True
    run.attrs["mask_rle_stale_reason"] = "test_edit"
    run.attrs["mask_rle_stale_row_count"] = 1

    assert open_mask_store(run, prefer="dense").encoding == "dense_uint8"
    with pytest.raises(ValueError, match="mask_rle is marked stale"):
        open_mask_store(run, prefer="rle")
    stale_store = open_mask_store(run, prefer="rle", allow_stale_rle=True)
    np.testing.assert_array_equal(stale_store.read_dense(), expected)

    dry_run = materialize_refined_subject_mask_store(zarr_path, apply=False, refresh_rle=True)
    summary = materialize_refined_subject_mask_store(zarr_path, apply=True, refresh_rle=True, chunk_size=1)

    assert dry_run["status"] == "would_refresh_rle"
    assert summary["status"] == "rle_refreshed"
    assert summary["rows"] == 3
    assert summary["has_dense_after"] is True
    assert summary["has_rle_after"] is True
    assert summary["mask_rle_stale_after"] is False

    reopened = zarr.open_group(str(zarr_path), mode="r")
    refreshed = reopened["refined_subject_masks_runs/refined_001"]
    mask_store = open_mask_store(refreshed, prefer="rle")
    np.testing.assert_array_equal(mask_store.read_dense(), updated)
    np.testing.assert_array_equal(np.asarray(refreshed["masks_roi"][:], dtype=np.uint8), updated)
    assert refreshed.attrs["mask_store_encodings"] == ["dense_uint8", "component_rle_v1"]
    assert refreshed.attrs["mask_storage_encoding"] == "dense_uint8+component_rle_v1"
    assert refreshed.attrs["mask_rle_stale"] is False
    assert "mask_rle_stale_reason" not in refreshed.attrs
    assert "mask_rle_stale_row_count" not in refreshed.attrs
    assert refreshed.attrs["mask_rle_refreshed_from"] == "masks_roi"


def test_materialize_refined_subject_mask_store_refreshes_only_selected_rle_component(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    expected, root = _build_compact_refined_zarr(zarr_path, keep_dense=True)
    run = root["refined_subject_masks_runs/refined_001"]
    untouched_group = run["mask_rle/components/02_eye_right"]
    untouched_before = {
        name: np.asarray(untouched_group[name][:])
        for name in ("counts", "indptr", "present", "area_px", "bbox_xyxy")
    }
    updated = expected.copy()
    updated[:, 1] = 0
    updated[0, 1, 1:9, 1:5] = 1
    run["masks_roi"][:] = updated
    mark_mask_rle_stale_attrs(
        run,
        reason="component_paint",
        updated_components=("eye_left",),
        updated_rows=(0,),
        updated_at_utc="2026-06-20T00:00:00+00:00",
    )

    summary = materialize_refined_subject_mask_store(
        zarr_path,
        apply=True,
        refresh_rle=True,
        components=("eye_left",),
        chunk_size=1,
    )

    assert summary["status"] == "rle_refreshed"
    assert summary["refresh_scope"] == "components"
    assert summary["refreshed_component_names"] == ["eye_left"]
    assert summary["refreshed_component_indices"] == [1]
    assert summary["mask_rle_stale_after"] is False
    reopened = zarr.open_group(str(zarr_path), mode="r")
    refreshed = reopened["refined_subject_masks_runs/refined_001"]
    refreshed_untouched = refreshed["mask_rle/components/02_eye_right"]
    for name, before in untouched_before.items():
        np.testing.assert_array_equal(np.asarray(refreshed_untouched[name][:]), before)
    assert refreshed.attrs["mask_rle_stale"] is False
    assert refreshed.attrs["mask_rle_refreshed_component_names"] == ["eye_left"]
    np.testing.assert_array_equal(open_mask_store(refreshed, prefer="rle").read_dense(), updated)


def test_materialize_refined_subject_mask_store_component_refresh_preserves_remaining_stale_scope(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    expected, root = _build_compact_refined_zarr(zarr_path, keep_dense=True)
    run = root["refined_subject_masks_runs/refined_001"]
    updated = expected.copy()
    updated[0, 1, 0:3, 0:3] = 1
    run["masks_roi"][:] = updated
    mark_mask_rle_stale_attrs(
        run,
        reason="multi_component_paint",
        updated_components=("eye_left", "eye_right"),
        updated_rows=(0, 1),
        updated_at_utc="2026-06-20T00:00:00+00:00",
    )

    summary = materialize_refined_subject_mask_store(
        zarr_path,
        apply=True,
        refresh_rle=True,
        components=("eye_left",),
        chunk_size=1,
    )

    assert summary["mask_rle_stale_after"] is True
    reopened = zarr.open_group(str(zarr_path), mode="r")
    refreshed = reopened["refined_subject_masks_runs/refined_001"]
    assert refreshed.attrs["mask_rle_stale"] is True
    assert refreshed.attrs["mask_rle_stale_component_names"] == ["eye_right"]
    assert refreshed.attrs["mask_rle_stale_row_count"] == 2
    with pytest.raises(ValueError, match="mask_rle is marked stale"):
        open_mask_store(refreshed, prefer="rle")
    diagnostic_store = open_mask_store(refreshed, prefer="rle", allow_stale_rle=True)
    diagnostic_dense = diagnostic_store.read_dense()
    np.testing.assert_array_equal(diagnostic_dense[:, 1], updated[:, 1])


def test_materialize_refined_subject_mask_store_refuses_to_delete_dense_when_rle_is_stale(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    _expected, root = _build_compact_refined_zarr(zarr_path, keep_dense=True)
    run = root["refined_subject_masks_runs/refined_001"]
    run.attrs["mask_rle_stale"] = True

    with pytest.raises(ValueError, match="compact mask_rle is marked stale"):
        materialize_refined_subject_mask_store(zarr_path, apply=True, delete_dense=True)

    reopened = zarr.open_group(str(zarr_path), mode="r")
    assert "masks_roi" in reopened["refined_subject_masks_runs/refined_001"]


def test_materialize_refined_subject_mask_store_deletes_dense_cache_when_compact_exists(tmp_path: Path) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    _expected, _root = _build_compact_refined_zarr(zarr_path, keep_dense=True)

    dry_run = materialize_refined_subject_mask_store(zarr_path, apply=False, delete_dense=True)
    summary = materialize_refined_subject_mask_store(zarr_path, apply=True, delete_dense=True)

    assert dry_run["status"] == "would_delete"
    assert summary["status"] == "deleted"
    reopened = zarr.open_group(str(zarr_path), mode="r")
    run = reopened["refined_subject_masks_runs/refined_001"]
    assert "masks_roi" not in run
    assert "mask_rle" in run
    assert run.attrs["masks_roi_materialized"] is False
    assert run.attrs["mask_store_encodings"] == ["component_rle_v1"]
    assert run.attrs["mask_storage_encoding"] == "component_rle_v1"


def test_materialize_refined_subject_mask_store_refuses_to_delete_last_store(tmp_path: Path) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    parent = root.create_group("refined_subject_masks_runs")
    parent.attrs["latest"] = "refined_001"
    run = parent.create_group("refined_001")
    run.attrs["mask_labels"] = ["subject_body"]
    run.create_array("masks_roi", data=np.ones((1, 1, 4, 4), dtype=np.uint8), overwrite=True)

    with pytest.raises(ValueError, match="Refusing to delete masks_roi"):
        materialize_refined_subject_mask_store(zarr_path, apply=True, delete_dense=True)
