"""Tests for merged-export registry linkage updates."""

import json
from pathlib import Path
import sys

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.utils.export_detect_training_zarr import _register_merged_dataset_in_registry


def test_register_merged_dataset_updates_training_set_linkage(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    merged_zarr = tmp_path / "merged_training.zarr"

    root = zarr.open_group(str(merged_zarr), mode="w")
    root.attrs["session_uuid"] = "detect_my_set_v001_merged"
    root.attrs["zarr_purpose"] = "training"
    raw = root.create_group("raw_video")
    raw.create_array(
        "images_ds",
        data=np.zeros((2, 8, 8), dtype=np.uint8),
        chunks=(1, 8, 8),
    )

    registry = Registry(registry_path)
    registry.upsert_training_set(
        set_id="detect_my_set_v001",
        name="my_set",
        query_filter={"tool": "fisheye.diagnostics.prepare_detect_training"},
        dataset_ids=["source_a", "source_b"],
        invocation={"tool": "fisheye.diagnostics.prepare_detect_training"},
    )
    registry.close()

    merged_dataset_id, linked = _register_merged_dataset_in_registry(
        registry_path=registry_path,
        merged_zarr=merged_zarr,
        set_id="detect_my_set_v001",
        set_name="my_set",
        source_dataset_ids=["source_a", "source_b"],
        query_filter={"tool": "fisheye.diagnostics.prepare_detect_training"},
        invocation={"tool": "fisheye.diagnostics.prepare_detect_training"},
    )

    assert merged_dataset_id == "detect_my_set_v001_merged"
    assert linked is True

    db = Registry(registry_path)
    dataset_row = db.conn.execute(
        "SELECT dataset_id, zarr_path FROM datasets WHERE dataset_id = ?",
        ("detect_my_set_v001_merged",),
    ).fetchone()
    assert dataset_row is not None
    assert str(merged_zarr) == dataset_row["zarr_path"]

    set_row = db.conn.execute(
        "SELECT dataset_ids_json FROM training_sets WHERE set_id = ?",
        ("detect_my_set_v001",),
    ).fetchone()
    assert set_row is not None
    dataset_ids = json.loads(set_row["dataset_ids_json"])
    assert dataset_ids == sorted(["source_a", "source_b", "detect_my_set_v001_merged"])
    db.close()
