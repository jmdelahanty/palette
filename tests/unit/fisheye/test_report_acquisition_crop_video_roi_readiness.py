from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from fisheye.utils.report_acquisition_crop_video_roi_readiness import (
    build_acquisition_crop_video_roi_readiness_report,
    main,
)


def _write_node(path: Path, *, attrs: dict | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    path.joinpath("zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": attrs or {},
            }
        ),
        encoding="utf-8",
    )


def _make_registry(path: Path) -> None:
    conn = sqlite3.connect(path)
    with conn:
        conn.execute(
            """
            CREATE TABLE datasets (
                dataset_id TEXT PRIMARY KEY,
                recording_id TEXT,
                zarr_path TEXT,
                zarr_use TEXT,
                status TEXT,
                artifact_kind TEXT
            );
            """
        )
    conn.close()


def _insert_dataset(
    path: Path,
    *,
    dataset_id: str,
    recording_id: str,
    zarr_path: Path,
    zarr_use: str,
) -> None:
    conn = sqlite3.connect(path)
    with conn:
        conn.execute(
            """
            INSERT INTO datasets (
                dataset_id, recording_id, zarr_path, zarr_use, status, artifact_kind
            )
            VALUES (?, ?, ?, ?, 'active', 'source_recording');
            """,
            (dataset_id, recording_id, str(zarr_path), zarr_use),
        )
    conn.close()


def _make_redscare_like_recording(tmp_path: Path) -> tuple[Path, Path, Path]:
    registry_path = tmp_path / "registry.sqlite"
    _make_registry(registry_path)

    recording_dir = tmp_path / "2026-06-23T16-01-09Z_arena_1_RedScare"
    zarr_dir = recording_dir / "zarr"
    analysis_zarr = zarr_dir / "2026-06-23T16-01-09Z_arena_1_RedScare_analysis.zarr"
    training_zarr = zarr_dir / "2026-06-23T16-01-09Z_arena_1_RedScare_training.zarr"

    _write_node(analysis_zarr)
    _write_node(analysis_zarr / "analysis" / "acquisition_video_streams")
    _write_node(analysis_zarr / "analysis" / "acquisition_video_streams" / "streams")
    _write_node(analysis_zarr / "analysis" / "acquisition_video_streams" / "streams" / "crop")

    _write_node(training_zarr)
    _write_node(training_zarr / "crop_runs", attrs={"latest": "crop_acq"})
    _write_node(training_zarr / "crop_runs" / "crop_acq", attrs={"crop_storage_mode": "materialized"})
    _write_node(training_zarr / "keypoints_runs", attrs={"latest": "keypoints_a"})
    _write_node(training_zarr / "keypoints_runs" / "keypoints_a")
    _write_node(training_zarr / "refined_keypoints_runs", attrs={"latest": "refined_keypoints_a"})
    _write_node(training_zarr / "refined_keypoints_runs" / "refined_keypoints_a")
    _write_node(training_zarr / "subject_mask_runs", attrs={"latest": "subject_masks_a"})
    _write_node(training_zarr / "subject_mask_runs" / "subject_masks_a")
    _write_node(training_zarr / "refined_subject_masks_runs", attrs={"latest": "refined_subject_masks_a"})
    _write_node(training_zarr / "refined_subject_masks_runs" / "refined_subject_masks_a")

    crop_dir = recording_dir / "derived" / "external_crop_recorder"
    crop_dir.mkdir(parents=True, exist_ok=True)
    crop_video = crop_dir / "Cam2010093_crop_external.mp4"
    crop_video.write_bytes(b"not a real mp4; report does not probe by default")
    crop_meta = crop_dir / "Cam2010093_crop_meta.csv"
    crop_meta.write_text(
        "\n".join(
            [
                "recording_frame_id,crop_video_frame_index,local_frame_id,has_detection,blank_frame,crop_x,crop_y,crop_w,crop_h",
                "1,0,10,1,0,100,200,384,384",
                "2,1,11,0,1,100,200,384,384",
                "3,2,12,1,0,100,200,0,384",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    recording_dir.joinpath("recording_manifest.json").write_text(
        json.dumps(
            {
                "video_streams": {
                    "streams": {
                        "crop": {
                            "video": "derived/external_crop_recorder/Cam2010093_crop_external.mp4",
                            "metadata": "derived/external_crop_recorder/Cam2010093_crop_meta.csv",
                            "width": 384,
                            "height": 384,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    _insert_dataset(
        registry_path,
        dataset_id="rec:analysis",
        recording_id="rec",
        zarr_path=analysis_zarr,
        zarr_use="analysis",
    )
    _insert_dataset(
        registry_path,
        dataset_id="rec:training",
        recording_id="rec",
        zarr_path=training_zarr,
        zarr_use="training",
    )
    return registry_path, analysis_zarr, training_zarr


def test_readiness_report_pairs_analysis_training_and_counts_crop_meta(tmp_path: Path) -> None:
    registry_path, _analysis_zarr, _training_zarr = _make_redscare_like_recording(tmp_path)

    report = build_acquisition_crop_video_roi_readiness_report(
        registry_path,
        path_contains="RedScare",
    )

    assert report["dataset_row_count"] == 2
    assert report["recording_count"] == 1
    assert report["action_counts"] == {"build_analysis_acquisition_crop_run": 1}
    row = report["records"][0]
    assert row["analysis_acquisition_crop_stream_present"] is True
    assert row["analysis_crop_runs_run_count"] == 0
    assert row["training_crop_runs_run_count"] == 1
    assert row["training_review_surfaces_present"] is True
    assert row["crop_width"] == 384
    assert row["crop_height"] == 384
    assert row["crop_video_meets_min_size"] is True
    assert row["crop_meta_rows"] == 3
    assert row["crop_meta_has_detection_rows"] == 2
    assert row["crop_meta_blank_rows"] == 1
    assert row["crop_meta_invalid_geometry_rows"] == 1
    assert row["crop_meta_usable_rows"] == 1


def test_readiness_report_cli_writes_jsonl(tmp_path: Path, capsys) -> None:
    registry_path, _analysis_zarr, _training_zarr = _make_redscare_like_recording(tmp_path)
    output_jsonl = tmp_path / "readiness.jsonl"

    assert (
        main(
            [
                "--registry",
                str(registry_path),
                "--path-contains",
                "RedScare",
                "--output-jsonl",
                str(output_jsonl),
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    assert "acquisition_crop_video_roi_readiness_report" in stdout
    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["record_type"] == "acquisition_crop_video_roi_readiness"
    assert rows[0]["recommended_next_action"] == "build_analysis_acquisition_crop_run"
