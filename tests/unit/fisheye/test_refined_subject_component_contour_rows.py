from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.shared.mask_store import write_component_rle_mask_store_from_dense
import fisheye.shared.refined_subject_component_contours as contour_mod
from fisheye.shared.refined_subject_component_contours import (
    build_component_contours_from_masks,
    refresh_component_contour_rows_from_masks,
    write_refined_subject_component_contours,
)


def _rectangle_mask(height: int, width: int, y0: int, x0: int, y1: int, x1: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[int(y0):int(y1), int(x0):int(x1)] = 1
    return mask


def _build_refined_run(store_path: Path) -> zarr.Group:
    root = zarr.open_group(str(store_path), mode="w")
    run = root.create_group("refined")
    run.attrs.update(
        {
            "run_name": "refined",
            "mask_labels": ["subject_body", "swim_bladder"],
            "label_schema_id": "subject_v1_lr",
        }
    )
    run.create_array("available_channels", data=np.asarray([True, True], dtype=bool), overwrite=True)
    masks = np.zeros((2, 2, 16, 16), dtype=np.uint8)
    masks[0, 0] = _rectangle_mask(16, 16, 2, 2, 12, 12)
    masks[1, 0] = _rectangle_mask(16, 16, 4, 4, 10, 10)
    masks[0, 1] = _rectangle_mask(16, 16, 6, 6, 9, 9)
    run.create_array("masks_roi", data=masks, chunks=(1, 1, 16, 16), overwrite=True)
    return run


def _decode_bytes_row(values: np.ndarray) -> str:
    raw = bytes(np.asarray(values, dtype=np.uint8).tolist())
    return raw.split(b"\x00", 1)[0].decode("utf-8")


def test_refresh_component_contour_row_appends_points_and_moves_pointer(tmp_path: Path) -> None:
    run = _build_refined_run(tmp_path / "contours.zarr")
    write_refined_subject_component_contours(
        run,
        components=["subject_body"],
        source_mask_run="refined",
        chunk_rois=1,
    )
    contours = run["components/subject_body/contours"]
    old_point_count = int(contours["points_xy"].shape[0])
    old_row_1_ptr = int(np.asarray(contours["ptr"][1], dtype=np.int64))

    masks = np.asarray(run["masks_roi"][:], dtype=np.uint8)
    masks[0, 0] = _rectangle_mask(16, 16, 1, 1, 5, 14)
    run["masks_roi"][:] = masks

    summaries = refresh_component_contour_rows_from_masks(
        run,
        "subject_body",
        [0],
        reason="unit_test_manual_edit",
        updated_at_utc="2026-04-29T12:00:00+00:00",
        chunk_rois=1,
    )

    assert len(summaries) == 1
    assert summaries[0].status == "written"
    assert summaries[0].point_offset == old_point_count
    assert int(contours["ptr"][0]) == old_point_count
    assert int(contours["ptr"][1]) == old_row_1_ptr
    assert int(contours["points_xy"].shape[0]) > old_point_count
    component = run["components/subject_body"]
    refreshed_contours = component["contours"]
    assert int(component["row_revision"][0]) == 1
    assert int(component["row_revision"][1]) == 0
    assert _decode_bytes_row(component["row_update_reason_bytes"][0]) == "unit_test_manual_edit"
    assert _decode_bytes_row(component["row_updated_at_utc_bytes"][0]) == "2026-04-29T12:00:00+00:00"
    assert refreshed_contours.attrs["cache_coverage"] == "full_indexed_rows"
    assert refreshed_contours.attrs["orphaned_points_possible"] is True


def test_write_component_contours_reads_compact_mask_store_without_dense_masks(tmp_path: Path) -> None:
    run = _build_refined_run(tmp_path / "compact_contours.zarr")
    dense = np.asarray(run["masks_roi"][:], dtype=np.uint8)
    write_component_rle_mask_store_from_dense(
        run,
        run["masks_roi"],
        component_names=tuple(str(value) for value in run.attrs["mask_labels"]),
        encode_row_chunk_size=1,
    )
    del run["masks_roi"]

    summaries = write_refined_subject_component_contours(
        run,
        components=["subject_body", "swim_bladder"],
        source_mask_run="refined",
        chunk_rois=1,
    )

    assert [summary.status for summary in summaries] == ["written", "written"]
    assert "masks_roi" not in run
    for component in ("subject_body", "swim_bladder"):
        contours = run[f"components/{component}/contours"]
        assert tuple(contours["ptr"].shape) == (2,)
        assert tuple(contours["len"].shape) == (2,)
        assert int(np.count_nonzero(np.asarray(contours["len"][:], dtype=np.int64) > 0)) == int(
            np.count_nonzero(dense[:, run.attrs["mask_labels"].index(component)].reshape(2, -1).any(axis=1))
        )


def test_build_component_contours_reads_masks_in_chunks(monkeypatch, tmp_path: Path) -> None:
    run = zarr.open_group(str(tmp_path / "chunked_contours.zarr"), mode="w")
    run.attrs["mask_labels"] = ["subject_body", "swim_bladder"]
    run.create_array("available_channels", data=np.asarray([True, True], dtype=bool), overwrite=True)

    dense = np.zeros((5, 2, 16, 16), dtype=np.uint8)
    for row_idx in range(5):
        dense[row_idx, 0] = _rectangle_mask(16, 16, 2, 2, 10, 10)

    class FakeMaskStore:
        n_rows = 5
        n_channels = 2
        shape = (5, 2, 16, 16)
        calls: list[tuple[slice | int | None, int]] = []

        def read_dense(self, *, rows=None, channels=None):
            self.calls.append((rows, int(channels)))
            if isinstance(rows, slice):
                return dense[rows, int(channels): int(channels) + 1]
            return dense[int(rows): int(rows) + 1, int(channels): int(channels) + 1]

    fake_store = FakeMaskStore()
    monkeypatch.setattr(contour_mod, "open_mask_store", lambda *_args, **_kwargs: fake_store)

    contours, summary = build_component_contours_from_masks(
        run,
        "subject_body",
        read_chunk_size=2,
    )

    assert summary.status == "computed"
    assert len(contours) == 5
    assert summary.contour_count == 5
    assert fake_store.calls == [
        (slice(0, 2), 0),
        (slice(2, 4), 0),
        (slice(4, 5), 0),
    ]


def test_refresh_component_contour_row_initializes_partial_cache_when_missing(tmp_path: Path) -> None:
    run = _build_refined_run(tmp_path / "partial.zarr")

    summaries = refresh_component_contour_rows_from_masks(
        run,
        "swim_bladder",
        [0],
        reason="unit_test_partial_cache",
        updated_at_utc="2026-04-29T12:00:00+00:00",
        chunk_rois=1,
    )

    assert summaries[0].status == "written"
    component = run["components/swim_bladder"]
    contours = component["contours"]
    assert contours.attrs["cache_coverage"] == "partial_row_updates"
    assert tuple(contours["ptr"].shape) == (2,)
    assert tuple(contours["len"].shape) == (2,)
    assert int(contours["ptr"][0]) == 0
    assert int(contours["len"][0]) > 0
    assert int(contours["ptr"][1]) == -1
    assert int(contours["len"][1]) == 0
    assert int(component["row_revision"][0]) == 1
    assert int(component["row_revision"][1]) == 0


def test_refresh_component_contour_row_records_missing_contour_for_empty_mask(tmp_path: Path) -> None:
    run = _build_refined_run(tmp_path / "empty.zarr")
    masks = np.asarray(run["masks_roi"][:], dtype=np.uint8)
    masks[0, 1] = 0
    run["masks_roi"][:] = masks

    summaries = refresh_component_contour_rows_from_masks(
        run,
        "swim_bladder",
        [0],
        reason="unit_test_empty_mask",
    )

    assert summaries[0].status == "missing_contour"
    component = run["components/swim_bladder"]
    contours = component["contours"]
    assert int(contours["ptr"][0]) == -1
    assert int(contours["len"][0]) == 0
    assert int(component["row_revision"][0]) == 1
