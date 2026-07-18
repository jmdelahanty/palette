import json
import sqlite3
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fisheye.utils.backfill_clipped_analysis_metadata import (
    backfill_clipped_analysis_metadata,
)


def _write_zarr_json(path: Path, attrs: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": attrs}),
        encoding="utf-8",
    )


def _read_zarr_attrs(path: Path) -> dict:
    return json.loads((path / "zarr.json").read_text(encoding="utf-8"))["attributes"]


def _write_frame_index(path: Path, *, old_root: Path) -> None:
    pq.write_table(
        pa.table(
            {
                "recording_folder": [str(old_root)],
                "camera_serial": ["2010095"],
                "clip_id": ["clip_000000"],
                "clip_local_frame_index": [0],
                "recording_frame_id": [1],
                "video_path": [str(old_root / "clips" / "clip_000000" / "Cam2010095.mp4")],
                "metadata_path": [str(old_root / "clips" / "clip_000000" / "Cam2010095_meta.csv")],
            }
        ),
        path,
    )


def _write_manifest(recording_root: Path, *, old_root: Path) -> None:
    (recording_root / "recording_frame_index_manifest.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "schema_version": "palette.recording_frame_index_manifest.v1",
                "frame_index_schema_version": "palette.recording_frame_index.v1",
                "recording_id": "sleepyfish_cam2010095",
                "session_id": "sleepyfish_cam2010095",
                "source_layout": "rolling_clips",
                "recording_frame_index_path": str(old_root / "recording_frame_index.parquet"),
                "row_count": 1,
                "recording_frame_id_min": 1,
                "recording_frame_id_max": 1,
            }
        ),
        encoding="utf-8",
    )


def _write_registry(path: Path, zarr_path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE datasets (
                dataset_id TEXT,
                recording_id TEXT,
                zarr_use TEXT,
                source_layout TEXT,
                source_recording_frame_index_path TEXT,
                source_frame_index_schema TEXT,
                zarr_path TEXT,
                last_seen_utc TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO datasets (
                dataset_id, recording_id, zarr_use, source_layout,
                source_recording_frame_index_path, source_frame_index_schema,
                zarr_path, last_seen_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "sleepyfish_cam2010095:ztest",
                None,
                "analysis",
                None,
                None,
                None,
                str(zarr_path.resolve()),
                "2026-01-01T00:00:00Z",
            ),
        )
        conn.commit()


def _make_archive(tmp_path: Path) -> tuple[Path, Path, Path]:
    recording_root = tmp_path / "sleepyfish_cam2010095"
    old_root = Path("/nvme1/recordings/sleepyfish_cam2010095")
    recording_root.mkdir()
    frame_index_path = recording_root / "recording_frame_index.parquet"
    _write_frame_index(frame_index_path, old_root=old_root)
    _write_manifest(recording_root, old_root=old_root)
    zarr_path = recording_root / "zarr" / "sleepyfish_cam2010095_analysis.zarr"
    old_video = old_root / "cams" / "Cam2010095.mp4"
    _write_zarr_json(
        zarr_path,
        {
            "zarr_purpose": "production",
            "source_path": str(old_video),
            "source_video_path": str(old_video),
            "source_video_metadata": {
                "codec": "hevc",
                "source_path": str(old_video),
            },
            "disk_path": str(old_root / "zarr" / zarr_path.name),
        },
    )
    _write_zarr_json(
        zarr_path / "raw_video",
        {
            "import_method": "metadata_only",
            "source_path": str(old_video),
        },
    )
    _write_zarr_json(zarr_path / "analysis_metadata", {})
    return zarr_path, recording_root, old_root


def test_backfill_clipped_analysis_metadata_plans_root_and_registry_changes(tmp_path: Path):
    zarr_path, _recording_root, _old_root = _make_archive(tmp_path)
    registry = tmp_path / "registry.sqlite"
    _write_registry(registry, zarr_path)

    result = backfill_clipped_analysis_metadata(zarr_path, registry_path=registry)

    assert result["status"] == "planned"
    assert result["attr_changes"]["zarr_purpose"]["wanted"] == "analysis"
    assert result["attr_changes"]["source_layout"]["wanted"] == "rolling_clips"
    assert result["registry"]["changes"]["source_layout"]["wanted"] == "rolling_clips"


def test_backfill_clipped_analysis_metadata_applies_root_registry_and_path_rewrite(tmp_path: Path):
    zarr_path, recording_root, old_root = _make_archive(tmp_path)
    registry = tmp_path / "registry.sqlite"
    _write_registry(registry, zarr_path)

    result = backfill_clipped_analysis_metadata(
        zarr_path,
        registry_path=registry,
        rewrite_frame_index_paths=True,
        old_root=str(old_root),
        new_root=str(recording_root),
        apply=True,
    )

    assert result["status"] == "applied"
    attrs = _read_zarr_attrs(zarr_path)
    assert attrs["zarr_purpose"] == "analysis"
    assert attrs["recording_id"] == "sleepyfish_cam2010095"
    assert attrs["source_layout"] == "rolling_clips"
    assert attrs["source_recording_frame_index_path"] == str(recording_root / "recording_frame_index.parquet")

    table = pq.read_table(recording_root / "recording_frame_index.parquet")
    assert table["video_path"][0].as_py().startswith(str(recording_root))
    assert table["metadata_path"][0].as_py().startswith(str(recording_root))

    with sqlite3.connect(registry) as conn:
        row = conn.execute(
            """
            SELECT recording_id, source_layout, source_recording_frame_index_path,
                   source_frame_index_schema
            FROM datasets
            WHERE zarr_path = ?
            """,
            (str(zarr_path.resolve()),),
        ).fetchone()
    assert row == (
        "sleepyfish_cam2010095",
        "rolling_clips",
        str(recording_root / "recording_frame_index.parquet"),
        "palette.recording_frame_index.v1",
    )


def test_backfill_repairs_live_video_locations_and_preserves_historical_provenance(
    tmp_path: Path,
):
    zarr_path, recording_root, old_root = _make_archive(tmp_path)
    source_video = recording_root / "cams" / "Cam2010095.mp4"
    source_video.parent.mkdir()
    source_video.write_bytes(b"video")

    planned = backfill_clipped_analysis_metadata(
        zarr_path,
        source_video_path=source_video,
    )
    assert planned["video_location"]["will_update"] is True
    assert set(planned["video_location"]["changes"]) == {
        "root.source_path",
        "root.source_video_path",
        "root.source_video_metadata.source_path",
        "raw_video.source_path",
    }
    assert _read_zarr_attrs(zarr_path)["source_path"].startswith("/nvme1/")

    result = backfill_clipped_analysis_metadata(
        zarr_path,
        source_video_path=source_video,
        apply=True,
    )
    assert result["video_location"]["updated"] is True

    wanted = str(source_video.resolve())
    root_attrs = _read_zarr_attrs(zarr_path)
    raw_attrs = _read_zarr_attrs(zarr_path / "raw_video")
    assert root_attrs["source_path"] == wanted
    assert root_attrs["source_video_path"] == wanted
    assert root_attrs["source_video_metadata"]["source_path"] == wanted
    assert raw_attrs["source_path"] == wanted

    assert root_attrs["source_video_metadata"]["codec"] == "hevc"
    assert root_attrs["disk_path"] == str(old_root / "zarr" / zarr_path.name)
    assert raw_attrs["import_method"] == "metadata_only"
    repair = root_attrs["source_video_location_repair"]
    assert repair["historical_environment_provenance_preserved"] is True
    assert repair["previous_live_fields"]["root.source_path"].startswith("/nvme1/")

    repeated = backfill_clipped_analysis_metadata(
        zarr_path,
        source_video_path=source_video,
        apply=True,
    )
    assert repeated["video_location"]["updated"] is False
    assert (
        _read_zarr_attrs(zarr_path)["source_video_location_repair"]
        == repair
    )


def test_backfill_rejects_missing_source_video(tmp_path: Path):
    zarr_path, recording_root, _old_root = _make_archive(tmp_path)

    try:
        backfill_clipped_analysis_metadata(
            zarr_path,
            source_video_path=recording_root / "cams" / "missing.mp4",
        )
    except FileNotFoundError as error:
        assert "Source video not found" in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected missing source video to be rejected")


def test_backfill_copies_dish_mask_and_experiment_setup_with_provenance(
    tmp_path: Path,
) -> None:
    zarr_path, _recording_root, _old_root = _make_archive(tmp_path)
    source_zarr = tmp_path / "source_training.zarr"
    dish_mask = {
        "shape": "circle",
        "detected_circle": {"center": [332, 326], "radius": 300},
        "metrics": {
            "image_shape": [640, 640],
            "center_norm": [0.51875, 0.509375],
            "radius_norm": 0.46875,
        },
    }
    experiment_setup = {
        "num_dishes": 1,
        "fish_per_dish": 1,
        "total_expected_fish": 1,
        "setup_type": "single_dish",
        "source": "video_only_intake_cli",
    }
    _write_zarr_json(source_zarr, {"experiment_setup": experiment_setup})
    _write_zarr_json(source_zarr / "analysis_metadata", {"dish_mask": dish_mask})

    planned = backfill_clipped_analysis_metadata(
        zarr_path,
        copy_analysis_metadata_from=source_zarr,
        require_dish_mask=True,
    )

    assert planned["analysis_context"]["dish_mask_will_update"] is True
    assert planned["analysis_context"]["experiment_setup_will_update"] is True
    assert "dish_mask" not in _read_zarr_attrs(zarr_path / "analysis_metadata")

    applied = backfill_clipped_analysis_metadata(
        zarr_path,
        copy_analysis_metadata_from=source_zarr,
        require_dish_mask=True,
        apply=True,
    )

    assert applied["analysis_context"]["updated"] is True
    analysis_attrs = _read_zarr_attrs(zarr_path / "analysis_metadata")
    root_attrs = _read_zarr_attrs(zarr_path)
    assert analysis_attrs["dish_mask"] == dish_mask
    assert analysis_attrs["dish_mask_source_zarr"] == str(source_zarr.resolve())
    assert analysis_attrs["dish_mask_source_key"] == "analysis_metadata.attrs.dish_mask"
    assert (
        analysis_attrs["dish_mask_copy_tool"]
        == "fisheye.utils.backfill_clipped_analysis_metadata"
    )
    assert root_attrs["experiment_setup"] == experiment_setup
    assert root_attrs["experiment_setup_source_zarr"] == str(source_zarr.resolve())
    assert (
        root_attrs["experiment_setup_copy_tool"]
        == "fisheye.utils.backfill_clipped_analysis_metadata"
    )

    repeated = backfill_clipped_analysis_metadata(
        zarr_path,
        copy_analysis_metadata_from=source_zarr,
        require_dish_mask=True,
        apply=True,
    )
    assert repeated["analysis_context"]["updated"] is False


def test_backfill_rejects_conflicting_existing_dish_mask(tmp_path: Path) -> None:
    zarr_path, _recording_root, _old_root = _make_archive(tmp_path)
    source_zarr = tmp_path / "source_training.zarr"
    _write_zarr_json(source_zarr, {"experiment_setup": {"setup_type": "single_dish"}})
    _write_zarr_json(
        source_zarr / "analysis_metadata",
        {"dish_mask": {"detected_circle": {"center": [1, 2], "radius": 3}}},
    )
    _write_zarr_json(
        zarr_path / "analysis_metadata",
        {"dish_mask": {"detected_circle": {"center": [4, 5], "radius": 6}}},
    )

    with pytest.raises(ValueError, match="conflicting dish_mask"):
        backfill_clipped_analysis_metadata(
            zarr_path,
            copy_analysis_metadata_from=source_zarr,
            require_dish_mask=True,
        )
