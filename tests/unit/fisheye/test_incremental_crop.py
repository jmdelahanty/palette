from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from fisheye.shared.keyed_delta import ACTION_CODE_MAP, REASON_CODE_MAP
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    RUN_STATUS_FAILED,
)
from fisheye.tracking.incremental_crop import (
    IncrementalCropError,
    materialize_incremental_crop_run,
)
from fisheye.utils.materialize_incremental_crop import (
    plan_or_materialize_incremental_crop,
)


PROVENANCE = {
    "schema": "palette.run_provenance.v1",
    "git_sha": "0" * 40,
    "config_hash": "1" * 64,
    "params": {"test": True},
    "input_run_ids": {"source": "test"},
    "input_artifacts": [],
    "command": "pytest incremental crop",
    "fisheye_version": None,
}


def _root() -> Any:
    root = zarr.group(store=zarr.storage.MemoryStore(), zarr_format=3)
    raw = root.create_group("raw_video")
    frames = np.stack(
        [
            np.arange(36, dtype=np.uint8).reshape(6, 6),
            np.arange(36, dtype=np.uint8).reshape(6, 6) + 50,
            np.arange(36, dtype=np.uint8).reshape(6, 6) + 100,
        ]
    )
    raw.create_array("images_full", data=frames, chunks=(1, 6, 6))
    return root


def _source(
    root: Any,
    name: str,
    *,
    keys: list[int],
    frames: list[int],
    boxes: list[list[float]],
) -> Any:
    group = root.require_group("refined_detect_runs").create_group(name)
    group.create_array("instance_key", data=np.asarray(keys, dtype=np.uint64))
    group.create_array("frame_indices", data=np.asarray(frames, dtype=np.int64))
    group.create_array("bbox_norm_coords", data=np.asarray(boxes, dtype=np.float32))
    group.attrs["edit_revision"] = 1
    return group


def _run(
    root: Any,
    source: Any,
    *,
    source_path: str,
    run_name: str,
    base_run_name: str | None = None,
    before_publish: Any = None,
) -> Any:
    return materialize_incremental_crop_run(
        root,
        source_group=source,
        source_path=source_path,
        frame_source=root["raw_video/images_full"],
        source_pixel_fingerprint="test-video-sha256",
        roi_size=(4, 4),
        run_name=run_name,
        run_provenance=PROVENANCE,
        base_run_name=base_run_name,
        roi_chunk_rows=2,
        signature_batch_rows=2,
        before_publish=before_publish,
    )


def test_initial_crop_materialization_publishes_complete_exact_payload() -> None:
    root = _root()
    source = _source(
        root,
        "source_a",
        keys=[11, 22],
        frames=[0, 1],
        boxes=[
            [0.5, 0.5, 0.2, 0.2],
            [0.0, 0.0, 0.2, 0.2],
        ],
    )

    result = _run(
        root,
        source,
        source_path="refined_detect_runs/source_a",
        run_name="crop_a",
    )

    crop = root["crop_runs/crop_a"]
    assert result.computed_rows == 2
    assert result.copied_rows == 0
    assert result.source_frame_bytes_read == 72
    assert crop.attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_COMPLETE
    assert root["crop_runs"].attrs["latest"] == "crop_a"
    assert root["crop_runs"].attrs["latest_materialized"] == "crop_a"
    np.testing.assert_array_equal(crop["instance_key"][:], [11, 22])
    np.testing.assert_array_equal(crop["roi_coordinates_full"][:], [[1, 1], [-2, -2]])
    expected_center = np.arange(36, dtype=np.uint8).reshape(6, 6)[1:5, 1:5]
    np.testing.assert_array_equal(crop["roi_images"][0], expected_center)
    expected_padded = np.zeros((4, 4), dtype=np.uint8)
    expected_padded[2:4, 2:4] = (
        np.arange(36, dtype=np.uint8).reshape(6, 6)[:2, :2] + 50
    )
    np.testing.assert_array_equal(crop["roi_images"][1], expected_padded)
    assert crop.attrs["materialization_summary"]["action_counts"]["compute"] == 2
    assert crop["materialization_plan"].attrs["schema_version"] == 1


def test_delta_crop_copies_unchanged_reordered_row_and_computes_only_changes() -> None:
    root = _root()
    source_a = _source(
        root,
        "source_a",
        keys=[11, 22, 33],
        frames=[0, 1, 2],
        boxes=[
            [0.5, 0.5, 0.2, 0.2],
            [0.5, 0.5, 0.2, 0.2],
            [0.5, 0.5, 0.2, 0.2],
        ],
    )
    _run(
        root,
        source_a,
        source_path="refined_detect_runs/source_a",
        run_name="crop_a",
    )
    source_b = _source(
        root,
        "source_b",
        keys=[33, 11, 44],
        frames=[2, 0, 1],
        boxes=[
            [0.25, 0.25, 0.2, 0.2],
            [0.5, 0.5, 0.2, 0.2],
            [0.75, 0.75, 0.2, 0.2],
        ],
    )
    source_b.create_array(
        "source_refined_row_ids",
        data=np.asarray([303, 101, 404], dtype=np.int64),
    )

    result = _run(
        root,
        source_b,
        source_path="refined_detect_runs/source_b",
        run_name="crop_b",
        base_run_name="crop_a",
    )

    crop_a = root["crop_runs/crop_a"]
    crop_b = root["crop_runs/crop_b"]
    assert result.copied_rows == 1
    assert result.computed_rows == 2
    assert result.omitted_rows == 1
    assert result.source_frame_bytes_read == 72
    np.testing.assert_array_equal(crop_b["instance_key"][:], [33, 11, 44])
    np.testing.assert_array_equal(crop_b["source_refined_row_ids"][:], [303, 101, 404])
    np.testing.assert_array_equal(crop_b["roi_images"][1], crop_a["roi_images"][0])
    np.testing.assert_array_equal(
        crop_b["materialization_plan/action_codes"][:],
        [ACTION_CODE_MAP["compute"], ACTION_CODE_MAP["copy"], ACTION_CODE_MAP["compute"]],
    )
    np.testing.assert_array_equal(
        crop_b["materialization_plan/reason_codes"][:],
        [REASON_CODE_MAP["source_changed"], REASON_CODE_MAP["unchanged"], REASON_CODE_MAP["added"]],
    )
    np.testing.assert_array_equal(
        crop_b["materialization_plan/omitted_instance_key"][:],
        [22],
    )


def test_pure_reorder_uses_no_source_frame_reads() -> None:
    root = _root()
    source_a = _source(
        root,
        "source_a",
        keys=[11, 22],
        frames=[0, 1],
        boxes=[[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]],
    )
    _run(
        root,
        source_a,
        source_path="refined_detect_runs/source_a",
        run_name="crop_a",
    )
    source_b = _source(
        root,
        "source_b",
        keys=[22, 11],
        frames=[1, 0],
        boxes=[[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]],
    )

    result = _run(
        root,
        source_b,
        source_path="refined_detect_runs/source_b",
        run_name="crop_b",
        base_run_name="crop_a",
    )

    assert result.copied_rows == 2
    assert result.computed_rows == 0
    assert result.source_frame_bytes_read == 0
    np.testing.assert_array_equal(
        root["crop_runs/crop_b/roi_images"][:],
        root["crop_runs/crop_a/roi_images"][:][::-1],
    )


def test_changed_pixel_contract_forces_full_compute_instead_of_reuse() -> None:
    root = _root()
    source = _source(
        root,
        "source_a",
        keys=[11, 22],
        frames=[0, 1],
        boxes=[[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]],
    )
    _run(
        root,
        source,
        source_path="refined_detect_runs/source_a",
        run_name="crop_a",
    )

    result = materialize_incremental_crop_run(
        root,
        source_group=source,
        source_path="refined_detect_runs/source_a",
        frame_source=root["raw_video/images_full"],
        source_pixel_fingerprint="different-video-sha256",
        roi_size=(4, 4),
        run_name="crop_b",
        run_provenance=PROVENANCE,
        base_run_name="crop_a",
        roi_chunk_rows=2,
        signature_batch_rows=2,
    )

    assert result.computed_rows == 2
    assert result.copied_rows == 0
    np.testing.assert_array_equal(
        root["crop_runs/crop_b/materialization_plan/reason_codes"][:],
        [REASON_CODE_MAP["signature_spec_changed"]] * 2,
    )


def test_source_change_during_processing_fails_without_replacing_latest() -> None:
    root = _root()
    source_a = _source(
        root,
        "source_a",
        keys=[11],
        frames=[0],
        boxes=[[0.5, 0.5, 0.2, 0.2]],
    )
    _run(
        root,
        source_a,
        source_path="refined_detect_runs/source_a",
        run_name="crop_a",
    )
    source_b = _source(
        root,
        "source_b",
        keys=[11, 22],
        frames=[0, 1],
        boxes=[[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]],
    )

    def mutate_source() -> None:
        source_b["bbox_norm_coords"][0, 0] = np.float32(0.25)

    with pytest.raises(IncrementalCropError, match="changed during processing"):
        _run(
            root,
            source_b,
            source_path="refined_detect_runs/source_b",
            run_name="crop_failed",
            base_run_name="crop_a",
            before_publish=mutate_source,
        )

    parent = root["crop_runs"]
    assert parent.attrs["latest"] == "crop_a"
    assert parent.attrs["latest_complete"] == "crop_a"
    assert parent.attrs["latest_materialized"] == "crop_a"
    assert parent["crop_failed"].attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_FAILED


def test_compute_failure_leaves_previous_complete_run_selected() -> None:
    root = _root()
    source_a = _source(
        root,
        "source_a",
        keys=[11],
        frames=[0],
        boxes=[[0.5, 0.5, 0.2, 0.2]],
    )
    _run(
        root,
        source_a,
        source_path="refined_detect_runs/source_a",
        run_name="crop_a",
    )
    source_b = _source(
        root,
        "source_b",
        keys=[11, 22],
        frames=[0, 1],
        boxes=[[0.5, 0.5, 0.2, 0.2], [0.75, 0.75, 0.2, 0.2]],
    )

    class FailingFrames:
        shape = root["raw_video/images_full"].shape
        dtype = root["raw_video/images_full"].dtype

        def __getitem__(self, index: Any) -> np.ndarray:
            raise OSError(f"injected frame read failure at {index}")

    with pytest.raises(OSError, match="injected frame read failure"):
        materialize_incremental_crop_run(
            root,
            source_group=source_b,
            source_path="refined_detect_runs/source_b",
            frame_source=FailingFrames(),
            source_pixel_fingerprint="test-video-sha256",
            roi_size=(4, 4),
            run_name="crop_failed",
            run_provenance=PROVENANCE,
            base_run_name="crop_a",
            roi_chunk_rows=2,
            signature_batch_rows=2,
        )

    parent = root["crop_runs"]
    assert parent.attrs["latest"] == "crop_a"
    assert parent.attrs["latest_complete"] == "crop_a"
    assert parent["crop_failed"].attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_FAILED


def test_newer_publication_is_not_overwritten_by_stale_materializer() -> None:
    root = _root()
    source_a = _source(
        root,
        "source_a",
        keys=[11],
        frames=[0],
        boxes=[[0.5, 0.5, 0.2, 0.2]],
    )
    _run(
        root,
        source_a,
        source_path="refined_detect_runs/source_a",
        run_name="crop_a",
    )
    source_b = _source(
        root,
        "source_b",
        keys=[11, 22],
        frames=[0, 1],
        boxes=[[0.5, 0.5, 0.2, 0.2], [0.75, 0.75, 0.2, 0.2]],
    )

    def publish_newer_run() -> None:
        parent = root["crop_runs"]
        newer = parent.create_group("crop_newer")
        newer.attrs[RUN_COMPLETION_STATUS_ATTR] = RUN_STATUS_COMPLETE
        parent.attrs.update(
            {
                "latest": "crop_newer",
                "latest_complete": "crop_newer",
                "latest_materialized": "crop_newer",
                "latest_any": "crop_newer",
                "publication_generation": 2,
            }
        )

    with pytest.raises(IncrementalCropError, match="publication state changed"):
        _run(
            root,
            source_b,
            source_path="refined_detect_runs/source_b",
            run_name="crop_stale",
            base_run_name="crop_a",
            before_publish=publish_newer_run,
        )

    parent = root["crop_runs"]
    assert parent.attrs["latest"] == "crop_newer"
    assert parent.attrs["latest_complete"] == "crop_newer"
    assert parent["crop_stale"].attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_FAILED


def test_legacy_crop_without_source_signatures_cannot_be_reused() -> None:
    root = _root()
    parent = root.create_group("crop_runs")
    legacy = parent.create_group("legacy")
    legacy.attrs.update(
        {
            RUN_COMPLETION_STATUS_ATTR: RUN_STATUS_COMPLETE,
            "crop_storage_mode": "materialized",
        }
    )
    legacy.create_array("instance_key", data=np.asarray([11], dtype=np.uint64))
    legacy.create_array("roi_images", data=np.zeros((1, 4, 4), dtype=np.uint8))
    source = _source(
        root,
        "source_a",
        keys=[11],
        frames=[0],
        boxes=[[0.5, 0.5, 0.2, 0.2]],
    )

    with pytest.raises(IncrementalCropError, match="not a Phase-1"):
        _run(
            root,
            source,
            source_path="refined_detect_runs/source_a",
            run_name="crop_a",
            base_run_name="legacy",
        )

    assert "crop_a" not in parent


def test_cli_dry_run_plans_without_creating_crop_parent(tmp_path: Path) -> None:
    archive = tmp_path / "analysis.zarr"
    root = zarr.open_group(str(archive), mode="w", zarr_format=3)
    raw = root.create_group("raw_video")
    raw.create_array(
        "images_full",
        data=np.arange(36, dtype=np.uint8).reshape(1, 6, 6),
    )
    source = root.create_group("refined_detect_runs").create_group("source_a")
    source.create_array("instance_key", data=np.asarray([11], dtype=np.uint64))
    source.create_array("frame_indices", data=np.asarray([0], dtype=np.int64))
    source.create_array(
        "bbox_norm_coords",
        data=np.asarray([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32),
    )

    report = plan_or_materialize_incremental_crop(
        archive,
        source_rowset_path="refined_detect_runs/source_a",
        source_pixel_fingerprint="test-video-sha256",
        roi_size=(4, 4),
        output_run="crop_a",
        base_crop_run=None,
        apply=False,
        roi_chunk_rows=2,
        signature_batch_rows=2,
        tabular_shard_rows=131_072,
        command="test dry run",
    )

    assert report["status"] == "would_materialize"
    assert report["plan"]["action_counts"]["compute"] == 1
    reopened = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    assert "crop_runs" not in reopened
