from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from fisheye.shared.source_video_metadata import build_source_video_metadata_v2
from fisheye.utils.preflight_source_video_metadata_backfill import (
    RegistryDataset,
    build_preflight_report,
    select_registry_datasets,
)


def _write_v3_attrs(group_path: Path, attrs: dict[str, object]) -> None:
    group_path.mkdir(parents=True, exist_ok=True)
    (group_path / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": attrs,
            }
        ),
        encoding="utf-8",
    )


def _make_dataset(
    tmp_path: Path,
    *,
    name: str = "recording_GoodCopBadCop",
    versioned: bool = False,
    root_overrides: dict[str, object] | None = None,
) -> RegistryDataset:
    recording = tmp_path / "recordings" / name
    video = recording / "cams" / "source.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    zarr_path = recording / "zarr" / f"{name}_analysis.zarr"
    metadata: dict[str, object] = {
        "source_path": str(video),
        "width": 4512,
        "height": 4512,
        "fps": 100.0,
        "total_frames": 1000,
    }
    if versioned:
        metadata = build_source_video_metadata_v2(
            metadata,
            recording_path=recording,
        )
    root_attrs: dict[str, object] = {
        "recording_path": str(recording),
        "source_video": video.name,
        "source_video_path": str(video),
        "source_path": str(video),
        "source_video_metadata": metadata,
        "width": 4512,
        "height": 4512,
        "fps": 100.0,
        "total_frames": 1000,
    }
    root_attrs.update(root_overrides or {})
    _write_v3_attrs(zarr_path, root_attrs)
    _write_v3_attrs(zarr_path / "raw_video", {"source_path": str(video)})
    return RegistryDataset(
        dataset_id=f"dataset:{name}",
        recording_id=name,
        zarr_path=zarr_path,
        dataset_status="active",
        zarr_use="analysis",
        artifact_kind="source_recording",
        source_layout=None,
        registry_recording_path=recording,
        recording_name=name,
    )


def _report(tmp_path: Path, datasets: list[RegistryDataset], *, expected: int):
    return build_preflight_report(
        datasets,
        registry_path=tmp_path / "registry.sqlite",
        path_contains="GoodCopBadCop",
        expected_count=expected,
        required_storage_root=tmp_path,
    )


def test_preflight_plans_recording_relative_v2_upgrade(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)

    report = _report(tmp_path, [dataset], expected=1)

    assert report["summary"]["ready_to_apply"] is True
    assert report["summary"]["dispositions"] == {"eligible": 1}
    row = report["rows"][0]
    assert row["errors"] == []
    assert row["planned_root_updates"]["source_video_metadata"]["locator"] == {
        "kind": "recording_relative",
        "relative_path": "cams/source.mp4",
    }


def test_preflight_recognizes_already_versioned_archive(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path, versioned=True)

    report = _report(tmp_path, [dataset], expected=1)

    assert report["summary"]["ready_to_apply"] is True
    assert report["summary"]["dispositions"] == {"already_v2": 1}


def test_preflight_blocks_conflicting_operational_paths(tmp_path: Path) -> None:
    dataset = _make_dataset(
        tmp_path,
        root_overrides={"source_video_path": str(tmp_path / "stale.mp4")},
    )

    report = _report(tmp_path, [dataset], expected=1)

    assert report["summary"]["ready_to_apply"] is False
    assert report["summary"]["dispositions"] == {"blocked": 1}
    assert any(
        error.startswith("source_video_resolution_failed:")
        for error in report["rows"][0]["errors"]
    )


def test_preflight_blocks_recording_root_mismatch(tmp_path: Path) -> None:
    dataset = _make_dataset(
        tmp_path,
        root_overrides={"recording_path": str(tmp_path / "wrong-recording")},
    )

    report = _report(tmp_path, [dataset], expected=1)

    assert report["summary"]["ready_to_apply"] is False
    assert "declared_recording_path_mismatch" in report["rows"][0]["errors"]


def test_preflight_fails_closed_on_cohort_count_or_duplicate_path(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    duplicate = RegistryDataset(
        **{
            **dataset.__dict__,
            "dataset_id": "duplicate-dataset",
            "recording_id": "duplicate-recording",
        }
    )

    report = _report(tmp_path, [dataset, duplicate], expected=1)

    assert report["summary"]["ready_to_apply"] is False
    assert "duplicate_zarr_paths" in report["summary"]["cohort_errors"]
    assert any(
        error.startswith("expected_count_mismatch:")
        for error in report["summary"]["cohort_errors"]
    )


def test_registry_selection_is_active_analysis_and_read_only(tmp_path: Path) -> None:
    registry = tmp_path / "registry.sqlite"
    conn = sqlite3.connect(registry)
    try:
        conn.executescript(
            """
            CREATE TABLE datasets (
                dataset_id TEXT PRIMARY KEY,
                recording_id TEXT,
                zarr_path TEXT NOT NULL,
                status TEXT,
                zarr_use TEXT,
                artifact_kind TEXT,
                source_layout TEXT
            );
            CREATE TABLE recordings (
                recording_id TEXT PRIMARY KEY,
                recording_path TEXT,
                recording_name TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO recordings VALUES (?, ?, ?)",
            ("rec-active", "/groups/recording", "rec-active"),
        )
        conn.executemany(
            "INSERT INTO datasets VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "active-analysis",
                    "rec-active",
                    "/groups/recording/GoodCopBadCop_analysis.zarr",
                    "active",
                    "analysis",
                    "source_recording",
                    None,
                ),
                (
                    "active-training",
                    "rec-active",
                    "/groups/recording/GoodCopBadCop_training.zarr",
                    "active",
                    "training",
                    "source_recording",
                    None,
                ),
                (
                    "deleted-analysis",
                    "rec-active",
                    "/groups/recording/old_GoodCopBadCop_analysis.zarr",
                    "deleted",
                    "analysis",
                    "source_recording",
                    None,
                ),
            ],
        )
        conn.commit()
    finally:
        conn.close()

    selected = select_registry_datasets(
        registry,
        path_contains="GoodCopBadCop",
    )

    assert [row.dataset_id for row in selected] == ["active-analysis"]
