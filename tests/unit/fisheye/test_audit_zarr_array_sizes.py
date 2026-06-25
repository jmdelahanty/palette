from __future__ import annotations

import json
from pathlib import Path

from fisheye.utils.audit_zarr_array_sizes import main, scan_zarr_array_sizes


def _write_group(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}}),
        encoding="utf-8",
    )


def _write_array(
    path: Path,
    *,
    shape: list[int],
    chunks: list[int],
    dtype: str = "uint8",
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "array",
                "shape": shape,
                "data_type": dtype,
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": chunks},
                },
                "chunk_key_encoding": {
                    "name": "default",
                    "configuration": {"separator": "/"},
                },
                "fill_value": 0,
                "codecs": [],
                "attributes": {},
            }
        ),
        encoding="utf-8",
    )


def _make_zarr(path: Path) -> Path:
    _write_group(path)
    _write_group(path / "refined_detect_runs")
    _write_group(path / "refined_detect_runs" / "run_a")
    _write_group(path / "refined_detect_runs" / "run_a" / "instances")
    _write_array(
        path / "refined_detect_runs" / "run_a" / "instances" / "bbox_xyxy",
        shape=[100, 4],
        chunks=[100, 4],
        dtype="float32",
    )
    _write_group(path / "refined_subject_masks_runs")
    _write_group(path / "refined_subject_masks_runs" / "run_a")
    _write_array(
        path / "refined_subject_masks_runs" / "run_a" / "masks_roi",
        shape=[100, 4, 512, 512],
        chunks=[16, 1, 512, 512],
        dtype="uint8",
    )
    _write_group(path / "refined_subject_masks_runs" / "run_a" / "components")
    _write_group(path / "refined_subject_masks_runs" / "run_a" / "components" / "eye_left")
    _write_group(
        path
        / "refined_subject_masks_runs"
        / "run_a"
        / "components"
        / "eye_left"
        / "contours"
    )
    _write_array(
        path
        / "refined_subject_masks_runs"
        / "run_a"
        / "components"
        / "eye_left"
        / "contours"
        / "points_xy",
        shape=[5000, 2],
        chunks=[1024, 2],
        dtype="float32",
    )
    return path


def test_scan_zarr_array_sizes_classifies_memory_and_write_strategy(tmp_path: Path) -> None:
    zarr_path = _make_zarr(tmp_path / "example_analysis.zarr")

    rows = scan_zarr_array_sizes(zarr_path, preload_threshold_bytes=1024 * 1024)
    by_path = {row.array_path: row for row in rows}

    bbox = by_path["refined_detect_runs/run_a/instances/bbox_xyxy"]
    assert bbox.logical_bytes == 100 * 4 * 4
    assert bbox.chunk_count == 1
    assert bbox.surface_family == "detection_geometry"
    assert bbox.memory_strategy == "preload_candidate"
    assert bbox.write_strategy == "preload_read_cache_with_row_or_overlay_writes"

    masks = by_path["refined_subject_masks_runs/run_a/masks_roi"]
    assert masks.logical_bytes == 100 * 4 * 512 * 512
    assert masks.chunk_count == 28
    assert masks.surface_family == "dense_masks"
    assert masks.memory_strategy == "lazy_chunked"
    assert masks.write_strategy == "chunked_surface_writes"

    contours = by_path[
        "refined_subject_masks_runs/run_a/components/eye_left/contours/points_xy"
    ]
    assert contours.surface_family == "ragged_mask_geometry"
    assert contours.memory_strategy == "preload_index_or_small_ragged"
    assert contours.write_strategy == "component_or_run_level_rewrite"


def test_scan_zarr_array_sizes_counts_physical_chunks_when_requested(tmp_path: Path) -> None:
    zarr_path = _make_zarr(tmp_path / "example_analysis.zarr")
    chunk_dir = zarr_path / "refined_subject_masks_runs" / "run_a" / "masks_roi" / "c"
    (chunk_dir / "0" / "0" / "0" / "0").parent.mkdir(parents=True, exist_ok=True)
    (chunk_dir / "0" / "0" / "0" / "0").write_bytes(b"abc")
    (chunk_dir / "1" / "0" / "0" / "0").parent.mkdir(parents=True, exist_ok=True)
    (chunk_dir / "1" / "0" / "0" / "0").write_bytes(b"defgh")

    rows = scan_zarr_array_sizes(zarr_path, collect_physical=True)
    masks = {
        row.array_path: row
        for row in rows
    }["refined_subject_masks_runs/run_a/masks_roi"]

    assert masks.physical_file_count == 2
    assert masks.physical_bytes == 8
    assert masks.compression_ratio_logical_to_physical == masks.logical_bytes / 8


def test_main_emits_jsonl(tmp_path: Path, capsys) -> None:
    zarr_path = _make_zarr(tmp_path / "example_analysis.zarr")

    assert main([str(zarr_path), "--format", "jsonl", "--sort", "path", "--top", "1"]) == 0

    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["array_path"] == "refined_detect_runs/run_a/instances/bbox_xyxy"
    assert payload["memory_strategy"] == "preload_candidate"
