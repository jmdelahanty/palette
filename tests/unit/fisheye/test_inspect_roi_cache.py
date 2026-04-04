from __future__ import annotations

import json
import os
from pathlib import Path

from fisheye.utils import inspect_roi_cache as mod


def _write_cache_store(
    root: Path,
    name: str,
    *,
    attrs: dict[str, object],
    data_files: int = 0,
) -> Path:
    store = root / name
    store.mkdir(parents=True)
    payload = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": attrs,
    }
    (store / "zarr.json").write_text(json.dumps(payload, sort_keys=True))
    for idx in range(data_files):
        (store / f"data_{idx:03d}.bin").write_bytes(b"1234")
    return store


def test_build_plan_reports_complete_and_incomplete_entries(tmp_path: Path) -> None:
    cache_root = tmp_path / "palette_roi_cache"
    cache_root.mkdir()
    _write_cache_store(
        cache_root,
        "complete_cache.zarr",
        attrs={
            "cache_complete": True,
            "cache_key": "abc123",
            "archive_path": "/tmp/archive.zarr",
            "crop_run_name": "crop_001",
            "source_crop_storage_mode": "geometry_only",
            "frame_source_kind": "source_video_path",
            "cache_layout_profile": "scratch_v1",
            "cache_write_backend_effective": "kvikio_gds",
            "cache_acceleration": "gpu",
            "cache_roi_chunk_len": 128,
            "total_rois": 22876,
            "roi_shape": [512, 512],
        },
        data_files=2,
    )
    _write_cache_store(
        cache_root,
        "incomplete_cache.zarr",
        attrs={
            "cache_complete": False,
            "cache_key": "def456",
            "crop_run_name": "crop_002",
        },
        data_files=1,
    )

    plan = mod._build_plan(
        cache_root,
        delete_incomplete=True,
        older_than_days=None,
    )

    assert plan.cache_root == str(cache_root)
    assert len(plan.entries) == 2
    by_name = {Path(entry.path).name: entry for entry in plan.entries}
    assert by_name["complete_cache.zarr"].status == "complete"
    assert by_name["complete_cache.zarr"].cache_write_backend_effective == "kvikio_gds"
    assert by_name["complete_cache.zarr"].cache_acceleration == "gpu"
    assert by_name["complete_cache.zarr"].total_rois == 22876
    assert by_name["complete_cache.zarr"].roi_shape == (512, 512)
    assert by_name["incomplete_cache.zarr"].status == "incomplete"
    assert plan.delete_paths == [str(cache_root / "incomplete_cache.zarr")]


def test_resolve_roi_cache_root_prefers_env_over_tmp(monkeypatch, tmp_path: Path) -> None:
    env_root = tmp_path / "env-cache-root"
    monkeypatch.setenv("PALETTE_ROI_CACHE_ROOT", str(env_root))

    resolved = mod._resolve_roi_cache_root(None)

    assert resolved == env_root.resolve()


def test_main_apply_deletes_selected_entries(tmp_path: Path, capsys) -> None:
    cache_root = tmp_path / "palette_roi_cache"
    cache_root.mkdir()
    keep = _write_cache_store(
        cache_root,
        "keep_cache.zarr",
        attrs={"cache_complete": True, "cache_key": "keep"},
    )
    delete = _write_cache_store(
        cache_root,
        "delete_cache.zarr",
        attrs={"cache_complete": False, "cache_key": "delete"},
    )

    exit_code = mod.main(
        [
            str(cache_root),
            "--delete-incomplete",
            "--apply",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert keep.exists()
    assert not delete.exists()
    assert "deleted=1" in captured.out


def test_main_json_emits_machine_readable_summary(tmp_path: Path, capsys) -> None:
    cache_root = tmp_path / "palette_roi_cache"
    cache_root.mkdir()
    _write_cache_store(
        cache_root,
        "cache_a.zarr",
        attrs={"cache_complete": True, "cache_key": "abc123"},
    )

    exit_code = mod.main([str(cache_root), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["cache_root"] == str(cache_root)
    assert len(payload["entries"]) == 1
    assert payload["entries"][0]["cache_key"] == "abc123"
    assert payload["delete_paths"] == []
