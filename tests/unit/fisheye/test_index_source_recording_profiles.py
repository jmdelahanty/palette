from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.utils import index_source_recording_profiles as mod


class _FakeArray:
    def __init__(self, payload: bytes) -> None:
        self._payload = np.frombuffer(payload, dtype=np.uint8)

    def __getitem__(self, _key):  # type: ignore[no-untyped-def]
        return self._payload


class _FakeRoot:
    def __init__(self, arrays: dict[str, _FakeArray]) -> None:
        self._arrays = arrays

    def __getitem__(self, key: str) -> _FakeArray:
        return self._arrays[key]


def _make_registry(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(
            """
            CREATE TABLE datasets (
                dataset_id TEXT PRIMARY KEY,
                session_uuid TEXT,
                zarr_path TEXT NOT NULL,
                path_hash TEXT,
                created_utc TEXT,
                last_seen_utc TEXT,
                status TEXT,
                recording_id TEXT,
                artifact_kind TEXT,
                zarr_origin TEXT,
                zarr_use TEXT
            );

            CREATE TABLE detection_data_profile (
                dataset_id TEXT NOT NULL,
                profile_run TEXT NOT NULL,
                profile_created_utc TEXT,
                updated_utc TEXT,
                frames_total INTEGER,
                frames_with_detections INTEGER,
                detections_total INTEGER,
                coverage_percent REAL
            );
            CREATE VIEW detection_data_profile_latest AS
            SELECT * FROM detection_data_profile;

            CREATE TABLE keypoint_data_profile (
                dataset_id TEXT NOT NULL,
                profile_run TEXT NOT NULL,
                profile_created_utc TEXT,
                updated_utc TEXT,
                rows_total INTEGER,
                rows_usable INTEGER,
                usable_rate REAL
            );
            CREATE VIEW keypoint_data_profile_latest AS
            SELECT * FROM keypoint_data_profile;

            CREATE TABLE eye_mask_data_profile (
                dataset_id TEXT NOT NULL,
                profile_run TEXT NOT NULL,
                profile_created_utc TEXT,
                updated_utc TEXT,
                rows_total INTEGER,
                rows_usable INTEGER,
                usable_rate REAL,
                pair_success_rate REAL
            );
            CREATE VIEW eye_mask_data_profile_latest AS
            SELECT * FROM eye_mask_data_profile;
            """
        )
        conn.execute(
            """
            INSERT INTO datasets (dataset_id, zarr_path, status, recording_id, artifact_kind, zarr_use)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            ("recA:training", "/tmp/recA_training.zarr", "active", "recA", "source_recording", "training"),
        )
        conn.execute(
            """
            INSERT INTO datasets (dataset_id, zarr_path, status, recording_id, artifact_kind, zarr_use)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            ("recA:analysis", "/tmp/recA_analysis.zarr", "active", "recA", "source_recording", "analysis"),
        )
        conn.execute(
            """
            INSERT INTO datasets (dataset_id, zarr_path, status, recording_id, artifact_kind, zarr_use)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            ("non_source", "/tmp/non_source.zarr", "active", "other", "merged_training", "training"),
        )

        conn.execute(
            """
            INSERT INTO detection_data_profile
            (dataset_id, profile_run, profile_created_utc, updated_utc, frames_total, frames_with_detections, detections_total, coverage_percent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            ("recA:training", "detect_profile_1", "2026-03-01T00:00:00Z", "2026-03-01T00:05:00Z", 100, 80, 120, 80.0),
        )
        conn.execute(
            """
            INSERT INTO keypoint_data_profile
            (dataset_id, profile_run, profile_created_utc, updated_utc, rows_total, rows_usable, usable_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            ("recA:training", "keypoint_profile_1", "2026-03-01T01:00:00Z", "2026-03-01T01:05:00Z", 120, 90, 0.75),
        )
        conn.execute(
            """
            INSERT INTO eye_mask_data_profile
            (dataset_id, profile_run, profile_created_utc, updated_utc, rows_total, rows_usable, usable_rate, pair_success_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            ("recA:analysis", "eye_profile_1", "2026-03-01T02:00:00Z", "2026-03-01T02:05:00Z", 90, 70, 0.777, 0.91),
        )
        conn.commit()
    finally:
        conn.close()


def test_collect_source_recording_profile_entries_filters_and_assigns_profiles(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _make_registry(registry_path)

    entries_all = mod.collect_source_recording_profile_entries(
        registry_path=registry_path,
        zarr_use_filter="all",
    )
    assert len(entries_all) == 2
    by_id = {entry.dataset_id: entry for entry in entries_all}
    assert len(by_id["recA:training"].detection_profiles) == 1
    assert len(by_id["recA:training"].keypoint_profiles) == 1
    assert len(by_id["recA:training"].eye_mask_profiles) == 0
    assert len(by_id["recA:analysis"].eye_mask_profiles) == 1

    entries_training = mod.collect_source_recording_profile_entries(
        registry_path=registry_path,
        zarr_use_filter="training",
    )
    assert [entry.dataset_id for entry in entries_training] == ["recA:training"]


def test_render_source_recording_profile_index_html_contains_run_tables(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _make_registry(registry_path)
    entries = mod.collect_source_recording_profile_entries(registry_path=registry_path, zarr_use_filter="all")
    output_html = tmp_path / "index.html"

    html = mod.render_source_recording_profile_index_html(
        entries=entries,
        registry_path=registry_path,
        zarr_use_filter="all",
        output_html=output_html,
        title="Source Profiles",
        thumb_width=220,
    )
    assert "Source Profiles" in html
    assert "recA:training" in html
    assert "detect_profile_1" in html
    assert "keypoint_profile_1" in html
    assert "eye_profile_1" in html
    assert "Detection Profiles" in html


def test_render_source_recording_profile_index_html_contains_clickable_filters(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _make_registry(registry_path)
    entries = mod.collect_source_recording_profile_entries(registry_path=registry_path, zarr_use_filter="all")
    output_html = tmp_path / "index.html"

    html = mod.render_source_recording_profile_index_html(
        entries=entries,
        registry_path=registry_path,
        zarr_use_filter="all",
        output_html=output_html,
        title="Source Profiles",
        thumb_width=220,
    )

    assert "data-filter-kind='profile'" in html
    assert "data-filter-kind='zarr-use'" in html
    assert "id='visible-count'" in html
    assert "data-has-detect='1'" in html
    assert "data-has-keypoint='1'" in html
    assert "data-has-eye-mask='1'" in html
    assert "data-zarr-use='training'" in html
    assert "data-zarr-use='analysis'" in html


def test_render_source_recording_profile_index_html_includes_artifact_thumbnail_links(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _make_registry(registry_path)
    entries = mod.collect_source_recording_profile_entries(registry_path=registry_path, zarr_use_filter="all")
    output_html = tmp_path / "index.html"
    artifacts_dir = tmp_path / "index.artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifacts_dir / "preview.png"
    artifact_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    by_id = {entry.dataset_id: entry for entry in entries}
    by_id["recA:training"].detection_profiles[0].artifact_png_paths.append(artifact_path)

    html = mod.render_source_recording_profile_index_html(
        entries=entries,
        registry_path=registry_path,
        zarr_use_filter="all",
        output_html=output_html,
        title="Source Profiles",
        thumb_width=180,
    )

    assert "artifact-strip" in html
    assert "preview.png" in html


def test_main_writes_html(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _make_registry(registry_path)
    output_html = tmp_path / "profiles_index.html"

    rc = mod.main(
        [
            "--registry",
            str(registry_path),
            "--output-html",
            str(output_html),
            "--zarr-use",
            "all",
        ]
    )
    assert rc == 0
    assert output_html.exists()
    text = output_html.read_text(encoding="utf-8")
    assert "Source Recording Profile Runs Index" in text
    assert "recA:training" in text


def test_enrich_entries_includes_detection_refined_run_artifacts(monkeypatch, tmp_path: Path) -> None:
    zarr_path = tmp_path / "rec_training.zarr"
    profile_run = "detection_profile_1"
    refined_run = "refined_detect_1"
    artifact_name = "detect_quality_overview_png"
    png_bytes = b"\x89PNG\r\n\x1a\nDETECT"

    profile_run_dir = zarr_path / "analysis" / "detection_profile_runs" / profile_run
    profile_run_dir.mkdir(parents=True, exist_ok=True)
    (profile_run_dir / "zarr.json").write_text(
        json.dumps(
            {
                "attributes": {
                    "profile_summary": {
                        "source": {
                            "detection_path": f"refined_detect_runs/{refined_run}/manual",
                            "refined_run": refined_run,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    vis_array_dir = zarr_path / "refined_detect_runs" / refined_run / "visualizations" / artifact_name
    vis_array_dir.mkdir(parents=True, exist_ok=True)
    (vis_array_dir / "zarr.json").write_text(json.dumps({"attributes": {}}), encoding="utf-8")

    array_path = f"refined_detect_runs/{refined_run}/visualizations/{artifact_name}"
    fake_root = _FakeRoot({array_path: _FakeArray(png_bytes)})
    monkeypatch.setattr(mod.zarr, "open_group", lambda *args, **kwargs: fake_root)

    row = mod.ProfileRow(
        profile_run=profile_run,
        profile_created_utc=None,
        updated_utc=None,
        metric_values=(),
    )
    entry = mod.SourceRecordingProfileEntry(
        dataset_id="recA:training",
        recording_id="recA",
        zarr_use="training",
        zarr_path=str(zarr_path),
        status="active",
        detection_profiles=[row],
    )
    artifacts_dir = tmp_path / "artifacts"
    written = mod.enrich_entries_with_profile_artifacts(
        [entry],
        artifacts_dir=artifacts_dir,
        overwrite=False,
    )

    assert written == 1
    assert len(row.artifact_png_paths) == 1
    assert row.artifact_png_paths[0].exists()
    assert row.artifact_png_paths[0].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_enrich_entries_includes_keypoint_refined_run_artifacts(monkeypatch, tmp_path: Path) -> None:
    zarr_path = tmp_path / "rec_training.zarr"
    profile_run = "keypoint_profile_1"
    refined_run = "refined_keypoints_1"
    artifact_name = "keypoint_quality_overview_png"
    png_bytes = b"\x89PNG\r\n\x1a\nKPT"

    profile_run_dir = zarr_path / "analysis" / "keypoint_profile_runs" / profile_run
    profile_run_dir.mkdir(parents=True, exist_ok=True)
    (profile_run_dir / "zarr.json").write_text(
        json.dumps(
            {
                "attributes": {
                    "profile_summary": {
                        "source": {
                            "keypoint_path": f"refined_keypoints_runs/{refined_run}",
                            "refined_run": refined_run,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    vis_array_dir = zarr_path / "refined_keypoints_runs" / refined_run / "visualizations" / artifact_name
    vis_array_dir.mkdir(parents=True, exist_ok=True)
    (vis_array_dir / "zarr.json").write_text(json.dumps({"attributes": {}}), encoding="utf-8")

    array_path = f"refined_keypoints_runs/{refined_run}/visualizations/{artifact_name}"
    fake_root = _FakeRoot({array_path: _FakeArray(png_bytes)})
    monkeypatch.setattr(mod.zarr, "open_group", lambda *args, **kwargs: fake_root)

    row = mod.ProfileRow(
        profile_run=profile_run,
        profile_created_utc=None,
        updated_utc=None,
        metric_values=(),
    )
    entry = mod.SourceRecordingProfileEntry(
        dataset_id="recA:training",
        recording_id="recA",
        zarr_use="training",
        zarr_path=str(zarr_path),
        status="active",
        keypoint_profiles=[row],
    )
    artifacts_dir = tmp_path / "artifacts"
    written = mod.enrich_entries_with_profile_artifacts(
        [entry],
        artifacts_dir=artifacts_dir,
        overwrite=False,
    )

    assert written == 1
    assert len(row.artifact_png_paths) == 1
    assert row.artifact_png_paths[0].exists()
    assert row.artifact_png_paths[0].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
