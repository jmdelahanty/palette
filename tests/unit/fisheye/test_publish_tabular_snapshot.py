from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr
from zarr.core.dtype import VariableLengthUTF8

from fisheye.shared.detect_reason_codec import decode_reason_bytes, write_reason_columns
from fisheye.shared.zarr_run_completion import (
    mark_run_complete,
    mark_run_started,
)
from fisheye.utils.publish_tabular_snapshot import publish_tabular_snapshot


def _complete_source(path: Path) -> tuple[Path, np.ndarray]:
    zarr_path = path / "sample.zarr"
    root = zarr.open_group(str(zarr_path), mode="w", zarr_format=3)
    parent = root.require_group("refined_keypoints_runs")
    source = parent.create_group("refined_source")
    mark_run_started(source, run_name="refined_source", stage="refined_keypoints")
    keys = np.arange(1, 21, dtype=np.uint64)
    source.create_array("instance_key", data=keys, chunks=(2,), overwrite=True)
    source.create_array(
        "keypoints_roi",
        data=np.arange(20 * 3 * 2, dtype=np.float64).reshape(20, 3, 2),
        chunks=(2, 3, 2),
        overwrite=True,
    )
    write_reason_columns(
        source,
        np.asarray(["clean"] * 20, dtype=object),
        chunk_size=2,
        overwrite=True,
    )
    legacy_reason = source.create_array(
        "reason",
        shape=(20,),
        chunks=(2,),
        dtype=VariableLengthUTF8(),
        fill_value="",
        overwrite=True,
    )
    legacy_reason[:] = np.asarray(["stale"] * 20, dtype=object)
    source.attrs["reason_fallback_order"] = ["reason_bytes", "reason", "detection_source"]
    source.create_array(
        "usable_keypoints",
        data=np.ones(20, dtype=bool),
        chunks=(2,),
        overwrite=True,
    )
    mark_run_complete(
        source,
        run_name="refined_source",
        allow_missing_run_provenance=True,
        missing_run_provenance_reason="unit fixture",
    )
    return zarr_path, keys


def test_publish_tabular_snapshot_is_exact_sharded_and_promoted(tmp_path: Path) -> None:
    zarr_path, keys = _complete_source(tmp_path)

    result = publish_tabular_snapshot(
        zarr_path=zarr_path,
        family="refined_keypoints_runs",
        source_run="refined_source",
        output_run="refined_snapshot",
        shard_rows=8,
        apply=True,
    )

    assert result["status"] == "complete"
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    parent = root["refined_keypoints_runs"]
    assert parent.attrs["latest"] == "refined_snapshot"
    target = parent["refined_snapshot"]
    assert target.attrs["artifact_mutability"] == "immutable_snapshot"
    assert target.attrs["snapshot_source_run"] == "refined_source"
    assert target["instance_key"].shards == (8,)
    assert target["keypoints_roi"].shards == (8, 3, 2)
    assert target["reason_bytes"].shards == (8, 64)
    assert "reason" not in target
    assert target.attrs["reason_authority"] == "reason_bytes"
    assert target.attrs["reason_fallback_order"] == ["reason_bytes", "detection_source"]
    assert result["omitted_legacy_array_paths"] == ["reason"]
    assert decode_reason_bytes(target["reason_bytes"][:]).tolist() == ["clean"] * 20
    np.testing.assert_array_equal(target["instance_key"][:], keys)
    np.testing.assert_array_equal(
        target["keypoints_roi"][:],
        parent["refined_source/keypoints_roi"][:],
    )


def test_publish_tabular_snapshot_defaults_to_dry_run(tmp_path: Path) -> None:
    zarr_path, _keys = _complete_source(tmp_path)

    result = publish_tabular_snapshot(
        zarr_path=zarr_path,
        family="refined_keypoints_runs",
        source_run="refined_source",
        output_run="refined_snapshot",
        shard_rows=8,
    )

    assert result["status"] == "planned"
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    assert "refined_snapshot" not in root["refined_keypoints_runs"]


def test_publish_tabular_snapshot_rejects_legacy_reason_without_reason_bytes(tmp_path: Path) -> None:
    zarr_path, _keys = _complete_source(tmp_path)
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    source = root["refined_keypoints_runs/refined_source"]
    del source["reason_bytes"]

    try:
        publish_tabular_snapshot(
            zarr_path=zarr_path,
            family="refined_keypoints_runs",
            source_run="refined_source",
            output_run="refined_snapshot",
            shard_rows=8,
        )
    except ValueError as exc:
        assert "canonicalize it to reason_bytes" in str(exc)
    else:
        raise AssertionError("Expected legacy-only reason publication to fail closed.")
