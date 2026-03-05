from pathlib import Path

import zarr

from fisheye.refinement import refine_keypoints as mod
from fisheye.registry.db import Registry


def _make_analysis_zarr(path: Path, *, session_uuid: str) -> zarr.Group:
    root = zarr.open_group(str(path), mode="w")
    root.attrs["session_uuid"] = session_uuid
    root.attrs["zarr_purpose"] = "analysis"
    return root


def test_resolve_status_context_from_root_uses_hashed_dataset_id_for_source_recordings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    monkeypatch.setenv("PALETTE_REGISTRY_PATH", str(registry_path))

    session_uuid = "2026-01-28T20-51-00Z_arena_1"
    zarr_path = tmp_path / "recordings" / "rec_a" / "zarr" / "rec_a_analysis.zarr"
    root = _make_analysis_zarr(zarr_path, session_uuid=session_uuid)

    context = mod._resolve_status_context_from_root(root, str(zarr_path))
    assert context is not None
    assert context.dataset_id.startswith(f"{session_uuid}:z")
    assert context.dataset_id != session_uuid
    assert context.recording_id == session_uuid

    registry = Registry(registry_path)
    try:
        rows = registry.conn.execute(
            "SELECT dataset_id FROM datasets WHERE zarr_path = ? ORDER BY dataset_id;",
            (str(zarr_path.resolve()),),
        ).fetchall()
        ids = [str(row["dataset_id"]) for row in rows]
        assert context.dataset_id in ids
        assert session_uuid not in ids
    finally:
        registry.close()


def test_resolve_status_context_from_root_prefers_hashed_dataset_when_legacy_id_exists(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    monkeypatch.setenv("PALETTE_REGISTRY_PATH", str(registry_path))

    session_uuid = "2026-01-28T20-51-00Z_arena_2"
    zarr_path = tmp_path / "recordings" / "rec_b" / "zarr" / "rec_b_analysis.zarr"
    _make_analysis_zarr(zarr_path, session_uuid=session_uuid)

    registry = Registry(registry_path)
    try:
        registry.upsert_dataset(
            session_uuid,
            session_uuid=session_uuid,
            zarr_path=zarr_path.resolve(),
            recording_id=session_uuid,
            zarr_use="analysis",
            zarr_purpose="analysis",
        )
        legacy_before = registry.conn.execute(
            "SELECT last_seen_utc FROM datasets WHERE dataset_id = ?;",
            (session_uuid,),
        ).fetchone()
        assert legacy_before is not None
        legacy_last_seen_before = str(legacy_before["last_seen_utc"])
    finally:
        registry.close()

    root = zarr.open_group(str(zarr_path), mode="a")
    context = mod._resolve_status_context_from_root(root, str(zarr_path))
    assert context is not None
    assert context.dataset_id.startswith(f"{session_uuid}:z")
    assert context.dataset_id != session_uuid

    registry = Registry(registry_path)
    try:
        legacy_after = registry.conn.execute(
            "SELECT last_seen_utc FROM datasets WHERE dataset_id = ?;",
            (session_uuid,),
        ).fetchone()
        assert legacy_after is not None
        assert str(legacy_after["last_seen_utc"]) == legacy_last_seen_before

        canonical = registry.conn.execute(
            "SELECT dataset_id, zarr_path FROM datasets WHERE dataset_id = ?;",
            (context.dataset_id,),
        ).fetchone()
        assert canonical is not None
        assert str(canonical["zarr_path"]) == str(zarr_path.resolve())
    finally:
        registry.close()
