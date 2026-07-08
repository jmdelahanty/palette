from __future__ import annotations

from pathlib import Path

import zarr

from fisheye.segmentation import infer_unet_subject_masks as mod
from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.shared.zarr_run_completion import RUN_COMPLETION_STATUS_ATTR, mark_run_complete


def _provenance(output_parent: str) -> dict[str, object]:
    return build_writer_run_provenance(
        command="test_infer_unet_subject_masks_shards",
        params={"output_parent": output_parent},
        input_run_ids={"crop": "crop_001"},
    )


def test_prepare_subject_mask_shard_run_does_not_touch_canonical_parent(tmp_path: Path) -> None:
    root = zarr.open_group(str(tmp_path / "archive.zarr"), mode="w")
    canonical_parent = root.require_group("subject_mask_runs")
    canonical_parent.attrs["latest"] = "canonical_subject_masks"
    canonical_parent.attrs["latest_complete"] = "canonical_subject_masks"
    canonical_parent.create_group("canonical_subject_masks")
    root.attrs["current_subject_mask_group_path"] = "subject_mask_runs/canonical_subject_masks"

    shard, run_name = mod._prepare_run_group(
        root,
        run_name="subject_mask_shard_clip_000001",
        overwrite=False,
        output_parent=mod.SUBJECT_MASK_SHARD_OUTPUT_PARENT,
    )
    shard.attrs.update(
        mod._shard_attrs_from_args(
            mod._build_arg_parser().parse_args(
                [
                    "archive.zarr",
                    "checkpoint.pt",
                    "--output-parent",
                    "subject_mask_shard_runs",
                    "--source-collection-id",
                    "collection_001",
                    "--source-clip-id",
                    "clip_000001",
                    "--source-clip-index",
                    "1",
                    "--source-roi-cache-alias-manifest",
                    "/tmp/cache_alias.json",
                    "--source-roi-cache-row-index-path",
                    "/tmp/cache_rows.parquet",
                ]
            ),
            output_parent=mod.SUBJECT_MASK_SHARD_OUTPUT_PARENT,
        )
    )
    mark_run_complete(
        shard,
        parent_group=root[mod.SUBJECT_MASK_SHARD_OUTPUT_PARENT],
        run_name=run_name,
        run_provenance=_provenance(mod.SUBJECT_MASK_SHARD_OUTPUT_PARENT),
    )

    assert root["subject_mask_runs"].attrs["latest"] == "canonical_subject_masks"
    assert root["subject_mask_runs"].attrs["latest_complete"] == "canonical_subject_masks"
    assert root.attrs["current_subject_mask_group_path"] == "subject_mask_runs/canonical_subject_masks"
    assert "subject_mask_shard_runs" in root
    assert root["subject_mask_shard_runs"].attrs["latest"] == "subject_mask_shard_clip_000001"
    assert root["subject_mask_shard_runs"].attrs["latest_complete"] == "subject_mask_shard_clip_000001"
    assert shard.attrs[RUN_COMPLETION_STATUS_ATTR] == "complete"
    assert shard.attrs["is_collection_shard"] is True
    assert shard.attrs["stage_selector_eligible"] is False
    assert shard.attrs["canonical_selector_publication"] == "suppressed_for_collection_shard"
    assert shard.attrs["source_collection_id"] == "collection_001"
    assert shard.attrs["source_clip_id"] == "clip_000001"
    assert shard.attrs["source_clip_index"] == 1
    assert shard.attrs["source_roi_cache_alias_manifest"] == "/tmp/cache_alias.json"
    assert shard.attrs["source_roi_cache_row_index_path"] == "/tmp/cache_rows.parquet"


def test_prepare_default_subject_mask_run_keeps_canonical_parent_behavior(tmp_path: Path) -> None:
    root = zarr.open_group(str(tmp_path / "archive.zarr"), mode="w")

    run, run_name = mod._prepare_run_group(
        root,
        run_name="subject_masks_full",
        overwrite=False,
    )
    mark_run_complete(
        run,
        parent_group=root["subject_mask_runs"],
        run_name=run_name,
        run_provenance=_provenance(mod.SUBJECT_MASK_CANONICAL_OUTPUT_PARENT),
    )

    assert "subject_mask_shard_runs" not in root
    assert root["subject_mask_runs"].attrs["latest"] == "subject_masks_full"
    assert root["subject_mask_runs"].attrs["latest_complete"] == "subject_masks_full"
    assert run.attrs[RUN_COMPLETION_STATUS_ATTR] == "complete"
