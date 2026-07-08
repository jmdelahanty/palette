from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

from fisheye.utils.create_clipped_collection_proxy_crop_run import (
    CLIPPED_COLLECTION_PROXY_DETECTION_SOURCE_TYPE,
)
from fisheye.utils.repair_clipped_proxy_crop_contract import (
    repair_clipped_proxy_crop_contract,
)


def _write_row_index(path: Path, *, clip_index: int, n_rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "bbox_norm_cx": np.full(n_rows, 0.25 + clip_index * 0.25, dtype=np.float32),
                "bbox_norm_cy": np.linspace(0.2, 0.4, n_rows, dtype=np.float32),
                "bbox_norm_w": np.full(n_rows, 0.1, dtype=np.float32),
                "bbox_norm_h": np.full(n_rows, 0.1, dtype=np.float32),
                "refined_detect_run": [f"refined_clip_{clip_index:06d}"] * n_rows,
                "refined_group_path": [
                    f"clips/clip_{clip_index:06d}/refined_detect_runs/refined_clip_{clip_index:06d}"
                ]
                * n_rows,
            }
        ),
        path,
    )


def _write_legacy_proxy(
    root: zarr.Group,
    name: str,
    *,
    frames: list[int],
    clip_index: int,
    row_index_path: Path,
) -> None:
    crop = root.require_group("crop_runs").create_group(name)
    frames_np = np.asarray(frames, dtype=np.int64)
    n_rows = int(frames_np.shape[0])
    crop.create_array("frame_indices", data=frames_np, chunks=(max(1, n_rows),))
    crop.attrs.update(
        {
            "source_kind": "finalized_clipped_refined_detect_collection_proxy",
            "source_collection_id": "collection_test",
            "source_clip_id": f"clip_{clip_index:06d}",
            "source_clip_index": clip_index,
            "source_roi_cache_row_index_path": str(row_index_path),
        }
    )


def test_repair_clipped_proxy_crop_contract_dry_apply_idempotent(tmp_path: Path) -> None:
    zarr_path = tmp_path / "sample_analysis.zarr"
    root = zarr.open_group(store=zarr_path, mode="w")
    row_index_path = tmp_path / "rows" / "clip_0.parquet"
    _write_row_index(row_index_path, clip_index=0, n_rows=2)
    _write_legacy_proxy(
        root,
        "crop_proxy_old",
        frames=[10, 11],
        clip_index=0,
        row_index_path=row_index_path,
    )

    dry = repair_clipped_proxy_crop_contract(zarr_path, apply=False)

    assert dry["status"] == "ok"
    assert dry["affected_crop_run_count"] == 1
    assert dry["changed_crop_run_count"] == 1
    assert dry["crop_runs"][0]["status"] == "would_update"
    assert "bbox_norm_coords" not in root["crop_runs/crop_proxy_old"]

    applied = repair_clipped_proxy_crop_contract(zarr_path, apply=True)

    assert applied["status"] == "ok"
    assert applied["changed_crop_run_count"] == 1
    crop = zarr.open_group(store=zarr_path, mode="r")["crop_runs/crop_proxy_old"]
    assert crop.attrs["detection_source_type"] == CLIPPED_COLLECTION_PROXY_DETECTION_SOURCE_TYPE
    assert crop.attrs["source_detect_run"] == f"{CLIPPED_COLLECTION_PROXY_DETECTION_SOURCE_TYPE}:collection_test"
    assert crop.attrs["source_detect_run_semantics"] == "synthetic_collection_rowset_label_not_detect_runs_child"
    assert crop.attrs["bbox_norm_coords_semantics"] == "bbox_xywh_normalized_to_full_frame"
    assert crop.attrs["source_refined_runs"] == ["refined_clip_000000"]
    np.testing.assert_allclose(
        crop["bbox_norm_coords"][:],
        np.asarray([[0.25, 0.2, 0.1, 0.1], [0.25, 0.4, 0.1, 0.1]], dtype=np.float32),
    )

    second = repair_clipped_proxy_crop_contract(zarr_path, apply=False)

    assert second["status"] == "ok"
    assert second["changed_crop_run_count"] == 0
    assert second["crop_runs"][0]["status"] == "ok"


def test_repair_clipped_proxy_crop_contract_repairs_merged_from_legacy_sources(tmp_path: Path) -> None:
    zarr_path = tmp_path / "sample_analysis.zarr"
    root = zarr.open_group(store=zarr_path, mode="w")
    crop_parent = root.require_group("crop_runs")

    row_index_a = tmp_path / "rows" / "clip_0.parquet"
    row_index_b = tmp_path / "rows" / "clip_1.parquet"
    _write_row_index(row_index_a, clip_index=0, n_rows=2)
    _write_row_index(row_index_b, clip_index=1, n_rows=2)
    _write_legacy_proxy(root, "crop_proxy_a", frames=[10, 11], clip_index=0, row_index_path=row_index_a)
    _write_legacy_proxy(root, "crop_proxy_b", frames=[20, 21], clip_index=1, row_index_path=row_index_b)

    merged = crop_parent.create_group("crop_proxy_collection")
    merged.create_array("frame_indices", data=np.asarray([10, 11, 20, 21], dtype=np.int64), chunks=(4,))
    merged.attrs.update(
        {
            "source_kind": "merged_clipped_collection_proxy_crop_run",
            "source_collection_id": "collection_test",
            "source_proxy_crop_runs": ["crop_proxy_a", "crop_proxy_b"],
        }
    )

    result = repair_clipped_proxy_crop_contract(
        zarr_path,
        crop_runs=["crop_proxy_collection"],
        apply=True,
    )

    assert result["status"] == "ok"
    assert result["changed_crop_run_count"] == 1
    repaired = zarr.open_group(store=zarr_path, mode="r")["crop_runs/crop_proxy_collection"]
    assert repaired.attrs["source_detect_run"] == f"{CLIPPED_COLLECTION_PROXY_DETECTION_SOURCE_TYPE}:collection_test"
    assert repaired.attrs["legacy_bbox_norm_coords_repair_count"] == 2
    assert repaired.attrs["source_refined_runs"] == ["refined_clip_000000", "refined_clip_000001"]
    np.testing.assert_allclose(
        repaired["bbox_norm_coords"][:],
        np.asarray(
            [
                [0.25, 0.2, 0.1, 0.1],
                [0.25, 0.4, 0.1, 0.1],
                [0.50, 0.2, 0.1, 0.1],
                [0.50, 0.4, 0.1, 0.1],
            ],
            dtype=np.float32,
        ),
    )
