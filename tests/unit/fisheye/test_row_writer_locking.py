"""Row writers serialize on the shared locks, across processes and in order."""

from __future__ import annotations

from contextlib import contextmanager
import multiprocessing
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.shared.zarr_helpers import (
    archive_metadata_publication_lock,
    archive_publication_lock_held_by_current_thread,
)
from fisheye.tune import refined_subject_mask_review as mask_review

ITERATIONS = 25


def _increment_row(archive: str, row: int, iterations: int) -> None:
    """Read-modify-write one row of a whole-run shard, as a row writer does."""

    for _ in range(iterations):
        with archive_metadata_publication_lock(archive):
            array = zarr.open_array(str(Path(archive) / "keypoints_roi"), mode="r+")
            value = np.asarray(array[row]).copy()
            array[row] = value + 1.0


def test_two_processes_editing_rows_of_one_shard_lose_no_updates(tmp_path: Path) -> None:
    archive = tmp_path / "review.zarr"
    zarr.create_array(
        str(archive / "keypoints_roi"),
        shape=(8, 19, 2),
        chunks=(1, 19, 2),
        shards=(8, 19, 2),  # one shard for the whole run, as the review profile writes
        dtype="float64",
        fill_value=0.0,
    )
    context = multiprocessing.get_context("spawn")
    workers = [
        context.Process(target=_increment_row, args=(str(archive), row, ITERATIONS))
        for row in (2, 5)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=120)
        assert worker.exitcode == 0
    array = zarr.open_array(str(archive / "keypoints_roi"), mode="r")
    np.testing.assert_array_equal(array[2], np.full((19, 2), ITERATIONS))
    np.testing.assert_array_equal(array[5], np.full((19, 2), ITERATIONS))
    for row in (0, 1, 3, 4, 6, 7):
        np.testing.assert_array_equal(array[row], np.zeros((19, 2)))


def test_run_lock_refuses_while_archive_lock_is_held(tmp_path: Path) -> None:
    archive = tmp_path / "review.zarr"
    archive.mkdir()
    with archive_metadata_publication_lock(archive):
        assert archive_publication_lock_held_by_current_thread(archive)
        with pytest.raises(RuntimeError, match="Lock order violation"):
            with mask_review._refined_subject_write_lock(archive, refined_run="run-a"):
                pass
    assert not archive_publication_lock_held_by_current_thread(archive)


def test_run_lock_then_archive_lock_is_the_permitted_order(tmp_path: Path) -> None:
    archive = tmp_path / "review.zarr"
    archive.mkdir()
    with mask_review._refined_subject_write_lock(archive, refined_run="run-a"):
        with archive_metadata_publication_lock(archive):
            assert archive_publication_lock_held_by_current_thread(archive)


@pytest.mark.parametrize(
    "helper",
    ["sync_refined_subject_mask_metadata", "check_refined_subject_source_updates"],
)
def test_maintenance_writers_hold_the_refined_run_lock(tmp_path: Path, monkeypatch, helper: str) -> None:
    held: list[tuple[str, str]] = []

    @contextmanager
    def recording_lock(zarr_path, *, refined_run, **_kwargs):
        held.append((str(zarr_path), str(refined_run)))
        yield {}

    monkeypatch.setattr(mask_review, "_refined_subject_write_lock", recording_lock)
    archive = tmp_path / "review.zarr"
    zarr.open_group(str(archive), mode="w")
    with pytest.raises(RuntimeError, match="not found"):
        getattr(mask_review, helper)(
            str(archive), refined_run="run-a", component_name="subject_body", roi_indices=[0],
        )
    assert held == [(str(archive), "run-a")]
