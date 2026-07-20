from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from fisheye.shared.composite_crop import (
    COMPOSITE_CROP_PAYLOAD_GROUP,
    CompositeCropArray,
    CompositeCropError,
    assert_crop_run_unreferenced,
    validate_composite_crop_run,
)
from fisheye.shared.crop_image_source import (
    CropImageSource,
)
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    RUN_STATUS_FAILED,
)
from fisheye.tracking.incremental_crop import (
    HISTORICAL_COMPOSITE_COORDINATE_CONTRACT,
    HISTORICAL_COMPOSITE_COORDINATE_CONTRACT_MODE,
    IncrementalCropError,
    materialize_composite_incremental_crop_run,
)
from tests.unit.fisheye.test_incremental_crop import _run as _canonical_test_run
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
    "command": "pytest composite crop",
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


def _standalone(
    root: Any,
    source: Any,
    *,
    source_name: str,
    run_name: str,
    base_run_name: str | None = None,
) -> Any:
    return _canonical_test_run(
        root,
        source,
        source_path=f"refined_detect_runs/{source_name}",
        run_name=run_name,
        base_run_name=base_run_name,
    )


def _composite(
    root: Any,
    source: Any,
    *,
    source_name: str,
    run_name: str,
    base_run_name: str,
    promote: bool = False,
    before_publish: Any = None,
) -> Any:
    return materialize_composite_incremental_crop_run(
        root,
        source_group=source,
        source_path=f"refined_detect_runs/{source_name}",
        frame_source=root["raw_video/images_full"],
        source_pixel_fingerprint="test-video-sha256",
        roi_size=(4, 4),
        run_name=run_name,
        run_provenance=PROVENANCE,
        base_run_name=base_run_name,
        roi_chunk_rows=2,
        signature_batch_rows=2,
        promote=promote,
        before_publish=before_publish,
        coordinate_contract_mode=HISTORICAL_COMPOSITE_COORDINATE_CONTRACT_MODE,
    )


def _base_and_target() -> tuple[Any, Any]:
    root = _root()
    source_a = _source(
        root,
        "source_a",
        keys=[11, 22, 33],
        frames=[0, 1, 2],
        boxes=[[0.5, 0.5, 0.2, 0.2]] * 3,
    )
    _standalone(root, source_a, source_name="source_a", run_name="crop_base")
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
    return root, source_b


def test_composite_writes_only_delta_and_reads_with_standalone_parity() -> None:
    root, source_b = _base_and_target()

    result = _composite(
        root,
        source_b,
        source_name="source_b",
        run_name="crop_composite",
        base_run_name="crop_base",
    )

    parent = root["crop_runs"]
    composite = parent["crop_composite"]
    payload = composite[COMPOSITE_CROP_PAYLOAD_GROUP]
    assert result.copied_rows == 1
    assert result.computed_rows == 2
    assert result.roi_payload_bytes_read_from_base == 0
    assert result.roi_payload_bytes_written == 2 * 4 * 4
    assert "roi_images" not in composite
    assert payload["roi_images_delta"].shape == (2, 4, 4)
    assert parent.attrs["latest"] == "crop_base"
    assert parent.attrs["latest_materialized"] == "crop_base"
    assert parent.attrs["latest_any"] == "crop_base"
    assert parent.attrs["latest_complete"] == "crop_base"
    assert composite.attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_COMPLETE
    assert composite.attrs["stage_selector_eligible"] is False
    assert composite.attrs["coordinate_contract"] == (
        HISTORICAL_COMPOSITE_COORDINATE_CONTRACT
    )

    source = CropImageSource.open(root, crop_run="crop_composite")
    assert source.storage_mode == "composite"
    assert source.roi_read_mode == "composite_base_delta"
    composite_pixels = source.read_indices([2, 1, 0, 1])

    _standalone(
        root,
        source_b,
        source_name="source_b",
        run_name="crop_expected",
        base_run_name="crop_base",
    )
    expected = np.asarray(root["crop_runs/crop_expected/roi_images"][:], dtype=np.uint8)
    np.testing.assert_array_equal(composite_pixels, expected[[2, 1, 0, 1]])
    np.testing.assert_array_equal(source.read_slice(0, 3), expected)


def test_composite_promotion_is_rejected_and_preserves_all_pointers() -> None:
    root, source_b = _base_and_target()

    with pytest.raises(IncrementalCropError, match="promotion is forbidden"):
        _composite(
            root,
            source_b,
            source_name="source_b",
            run_name="crop_composite",
            base_run_name="crop_base",
            promote=True,
        )

    attrs = root["crop_runs"].attrs
    assert attrs["latest"] == "crop_base"
    assert attrs["latest_materialized"] == "crop_base"
    assert attrs["latest_any"] == "crop_base"
    assert attrs["latest_complete"] == "crop_base"
    assert "latest_composite" not in attrs
    assert "crop_composite" not in root["crop_runs"]


def test_pure_reorder_composite_has_empty_delta_and_no_frame_reads() -> None:
    root = _root()
    source_a = _source(
        root,
        "source_a",
        keys=[11, 22],
        frames=[0, 1],
        boxes=[[0.5, 0.5, 0.2, 0.2]] * 2,
    )
    _standalone(root, source_a, source_name="source_a", run_name="crop_base")
    source_b = _source(
        root,
        "source_b",
        keys=[22, 11],
        frames=[1, 0],
        boxes=[[0.5, 0.5, 0.2, 0.2]] * 2,
    )

    result = _composite(
        root,
        source_b,
        source_name="source_b",
        run_name="crop_composite",
        base_run_name="crop_base",
    )

    assert result.computed_rows == 0
    assert result.source_frame_bytes_read == 0
    assert root[
        f"crop_runs/crop_composite/{COMPOSITE_CROP_PAYLOAD_GROUP}/roi_images_delta"
    ].shape == (0, 4, 4)
    source = CropImageSource.open(root, crop_run="crop_composite")
    np.testing.assert_array_equal(
        source[:],
        np.asarray(root["crop_runs/crop_base/roi_images"][:], dtype=np.uint8)[::-1],
    )


def test_composite_validation_fails_closed_on_mapping_corruption() -> None:
    root, source_b = _base_and_target()
    _composite(
        root,
        source_b,
        source_name="source_b",
        run_name="crop_composite",
        base_run_name="crop_base",
    )
    parent = root["crop_runs"]
    composite = parent["crop_composite"]
    composite[f"{COMPOSITE_CROP_PAYLOAD_GROUP}/source_row_indices"][1] = 999

    with pytest.raises(CompositeCropError, match="base row out of bounds"):
        validate_composite_crop_run(
            parent,
            composite,
            run_name="crop_composite",
            verify_identity=True,
        )


def test_composite_cannot_be_used_as_another_composite_base() -> None:
    root, source_b = _base_and_target()
    _composite(
        root,
        source_b,
        source_name="source_b",
        run_name="crop_composite",
        base_run_name="crop_base",
    )

    with pytest.raises(IncrementalCropError, match="not a Phase-1"):
        _composite(
            root,
            source_b,
            source_name="source_b",
            run_name="crop_depth_two",
            base_run_name="crop_composite",
        )
    assert "crop_depth_two" not in root["crop_runs"]


def test_composite_retains_base_against_deletion() -> None:
    root, source_b = _base_and_target()
    _composite(
        root,
        source_b,
        source_name="source_b",
        run_name="crop_composite",
        base_run_name="crop_base",
    )
    parent = root["crop_runs"]

    with pytest.raises(CompositeCropError, match="composite dependents"):
        assert_crop_run_unreferenced(parent, "crop_base")
    del parent["crop_composite"]
    assert_crop_run_unreferenced(parent, "crop_base")


def test_source_mutation_fails_without_selecting_composite() -> None:
    root, source_b = _base_and_target()

    def mutate_source() -> None:
        source_b["bbox_norm_coords"][0, 0] = np.float32(0.4)

    with pytest.raises(IncrementalCropError, match="changed during processing"):
        _composite(
            root,
            source_b,
            source_name="source_b",
            run_name="crop_failed",
            base_run_name="crop_base",
            before_publish=mutate_source,
        )
    parent = root["crop_runs"]
    assert parent.attrs["latest"] == "crop_base"
    assert parent.attrs["latest_any"] == "crop_base"
    assert parent["crop_failed"].attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_FAILED


class _CountingArray:
    def __init__(self, values: np.ndarray) -> None:
        self.values = np.asarray(values, dtype=np.uint8)
        self.shape = self.values.shape
        self.dtype = self.values.dtype
        self.reads: list[object] = []

    def __getitem__(self, key: object) -> np.ndarray:
        self.reads.append(key)
        return self.values[key]


def test_composite_array_reads_only_requested_source_ranges() -> None:
    base = _CountingArray(np.arange(10 * 2 * 2, dtype=np.uint8).reshape(10, 2, 2))
    delta = _CountingArray(
        (np.arange(3 * 2 * 2, dtype=np.uint8) + 100).reshape(3, 2, 2)
    )
    array = CompositeCropArray(
        base_array=base,
        delta_array=delta,
        source_codes=np.asarray([0, 1, 0, 1, 0], dtype=np.uint8),
        source_row_indices=np.asarray([8, 2, 1, 0, 2], dtype=np.int64),
        roi_shape=(2, 2),
    )

    result = array.read_indices([4, 1, 2, 4])

    np.testing.assert_array_equal(result[0], base.values[2])
    np.testing.assert_array_equal(result[1], delta.values[2])
    np.testing.assert_array_equal(result[2], base.values[1])
    assert base.reads == [slice(1, 3, None)]
    assert delta.reads == [slice(2, 3, None)]


def test_cli_composite_dry_run_reports_delta_bytes_without_mutation(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "analysis.zarr"
    root = zarr.open_group(str(archive), mode="w", zarr_format=3)
    raw = root.create_group("raw_video")
    raw.create_array(
        "images_full",
        data=np.arange(36, dtype=np.uint8).reshape(1, 6, 6),
    )
    source = _source(
        root,
        "source_a",
        keys=[11],
        frames=[0],
        boxes=[[0.5, 0.5, 0.2, 0.2]],
    )
    _standalone(root, source, source_name="source_a", run_name="crop_base")

    report = plan_or_materialize_incremental_crop(
        archive,
        source_rowset_path="refined_detect_runs/source_a",
        source_pixel_fingerprint="test-video-sha256",
        roi_size=(4, 4),
        output_run="crop_composite",
        base_crop_run="crop_base",
        apply=False,
        roi_chunk_rows=2,
        signature_batch_rows=2,
        tabular_shard_rows=131_072,
        command="test composite dry run",
        payload_strategy="composite",
        historical_composite=True,
    )

    assert report["payload_strategy"] == "composite"
    assert report["estimated_roi_payload_bytes"] == 0
    reopened = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    assert "crop_composite" not in reopened["crop_runs"]
