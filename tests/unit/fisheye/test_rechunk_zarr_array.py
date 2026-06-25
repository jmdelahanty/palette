from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import zarr

from fisheye.utils.rechunk_zarr_array import main, rechunk_zarr_array


def _make_store(path: Path) -> tuple[Path, str, np.ndarray]:
    zarr_path = path / "sample.zarr"
    root = zarr.open_group(str(zarr_path), mode="w", zarr_format=3)
    run = root.require_group("arena_assignment_runs").require_group("run_a")
    data = np.arange(25, dtype=np.int32).reshape(25, 1)
    arr = run.create_array(
        "n_detections_per_arena",
        data=data,
        chunks=(5, 1),
        overwrite=True,
    )
    arr.attrs["source"] = "unit-test"
    return zarr_path, "arena_assignment_runs/run_a/n_detections_per_arena", data


def test_rechunk_zarr_array_dry_run_does_not_modify_store(tmp_path: Path) -> None:
    zarr_path, array_path, _data = _make_store(tmp_path)

    summary = rechunk_zarr_array(zarr_path, array_path, row_chunk=16, apply=False)

    assert summary.status == "planned"
    assert summary.old_chunks == (5, 1)
    assert summary.new_chunks == (16, 1)
    assert summary.old_chunk_count == 5
    assert summary.new_chunk_count == 2
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    assert root[array_path].chunks == (5, 1)


def test_rechunk_zarr_array_applies_and_preserves_data_attrs(tmp_path: Path) -> None:
    zarr_path, array_path, data = _make_store(tmp_path)

    summary = rechunk_zarr_array(
        zarr_path,
        array_path,
        row_chunk=16,
        storage_profile_id="geometry_preload_v1_canary",
        reason="unit test",
        apply=True,
    )

    assert summary.status == "updated"
    assert summary.applied is True
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    arr = root[array_path]
    assert arr.chunks == (16, 1)
    np.testing.assert_array_equal(arr[:], data)
    assert arr.attrs["source"] == "unit-test"
    assert arr.attrs["storage_profile_id"] == "geometry_preload_v1_canary"
    assert arr.attrs["rechunk_provenance"]["old_chunk_shape"] == [5, 1]
    assert arr.attrs["rechunk_provenance"]["new_chunk_shape"] == [16, 1]
    assert arr.attrs["rechunk_provenance"]["reason"] == "unit test"
    assert "n_detections_per_arena__rechunk_tmp" not in set(root["arena_assignment_runs/run_a"].keys())


def test_main_emits_json_summary(tmp_path: Path, capsys) -> None:
    zarr_path, array_path, _data = _make_store(tmp_path)

    assert main([str(zarr_path), array_path, "--row-chunk", "16", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "planned"
    assert payload["old_chunks"] == [5, 1]
    assert payload["new_chunks"] == [16, 1]
