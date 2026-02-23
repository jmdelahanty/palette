from __future__ import annotations

from pathlib import Path

import zarr

from fisheye.registry.db import Registry
from fisheye.utils import generate_review_list as mod


def _make_crop_zarr(
    path: Path,
    *,
    run_status: str,
    review_state: str | None = None,
) -> None:
    root = zarr.open_group(str(path), mode="w")
    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_2026-02-10_10-00-00"
    crop_run = crop_parent.create_group("crop_2026-02-10_10-00-00")
    crop_run.attrs["status"] = run_status
    if review_state is not None:
        crop_run.attrs["crop_review_status"] = {"state": review_state}


def _seed_crop_registry(registry_path: Path, rows: list[dict[str, object]]) -> None:
    registry = Registry(registry_path)
    for row in rows:
        dataset_id = str(row["dataset_id"])
        zarr_path = Path(str(row["zarr_path"]))
        recording_id = str(row["recording_id"])
        registry.upsert_dataset(
            dataset_id,
            session_uuid=f"session_{dataset_id}",
            zarr_path=zarr_path,
            recording_id=recording_id,
            artifact_kind="source_recording",
            zarr_use="analysis",
        )
        registry.upsert_provenance(
            dataset_id,
            provenance={},
            context={},
            protocol_name=None,
            protocol_hash=None,
            acquisition={},
            zarr_purpose="analysis",
        )
        registry.replace_crop_quality(
            dataset_id,
            [
                {
                    "crop_run": str(row["crop_run"]),
                    "recording_id": recording_id,
                    "zarr_use": "analysis",
                    "crop_created_utc": "2026-02-10T00:00:00+00:00",
                    "source_detect_run": None,
                    "source_refined_run": "refined_detect_2026-02-10_00-00-00",
                    "detection_source_type": "manual",
                    "detection_source_path": "refined_detect_runs/refined_detect_2026-02-10_00-00-00/manual",
                    "total_rois": 100,
                    "frames_with_crops": 90,
                    "total_frames": 100,
                    "percent_frames_with_crops": 90.0,
                    "includes_interpolated": 0,
                    "n_real_detections": 100,
                    "n_interpolated_detections": 0,
                    "review_state": row.get("review_state"),
                    "review_method": "manual",
                    "review_intended_use": row.get("review_intended_use"),
                    "review_reviewer": "tester",
                    "review_timestamp_utc": "2026-02-10T00:10:00+00:00",
                    "review_notes": None,
                    "zarr_mtime_ns": int(row.get("zarr_mtime_ns", zarr_path.stat().st_mtime_ns)),
                    "updated_utc": "2026-02-10T00:10:00+00:00",
                }
            ],
        )
    registry.close()


def test_generate_review_list_crop_defaults_to_completed_and_missing(tmp_path: Path) -> None:
    rec_root = tmp_path / "recordings"
    rec_root.mkdir()

    zarr_a = rec_root / "a_analysis.zarr"
    zarr_b = rec_root / "b_analysis.zarr"
    zarr_c = rec_root / "c_analysis.zarr"
    _make_crop_zarr(zarr_a, run_status="completed", review_state=None)
    _make_crop_zarr(zarr_b, run_status="running", review_state=None)
    _make_crop_zarr(zarr_c, run_status="completed", review_state="approved")

    output = tmp_path / "crop_review_list.txt"
    rc = mod.main(
        [
            str(rec_root),
            "--recursive",
            "--stage",
            "crop",
            "--output",
            str(output),
        ]
    )
    assert rc == 0

    lines = [line.strip() for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines == [str(zarr_a.resolve())]


def test_generate_review_list_crop_status_any_includes_running(tmp_path: Path) -> None:
    rec_root = tmp_path / "recordings"
    rec_root.mkdir()

    zarr_a = rec_root / "a_analysis.zarr"
    zarr_b = rec_root / "b_analysis.zarr"
    _make_crop_zarr(zarr_a, run_status="completed", review_state=None)
    _make_crop_zarr(zarr_b, run_status="running", review_state=None)

    output = tmp_path / "crop_review_list.txt"
    rc = mod.main(
        [
            str(rec_root),
            "--recursive",
            "--stage",
            "crop",
            "--crop-run-status",
            "any",
            "--output",
            str(output),
        ]
    )
    assert rc == 0

    lines = [line.strip() for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines == sorted([str(zarr_a.resolve()), str(zarr_b.resolve())])


def test_generate_review_list_crop_registry_mode_respects_completed_filter(tmp_path: Path) -> None:
    rec_root = tmp_path / "recordings"
    rec_root.mkdir()
    zarr_a = rec_root / "a_analysis.zarr"
    zarr_b = rec_root / "b_analysis.zarr"
    _make_crop_zarr(zarr_a, run_status="completed", review_state=None)
    _make_crop_zarr(zarr_b, run_status="running", review_state=None)

    registry_path = tmp_path / "registry.sqlite"
    _seed_crop_registry(
        registry_path,
        rows=[
            {
                "dataset_id": "dataset_a",
                "zarr_path": zarr_a,
                "recording_id": "recording_a",
                "crop_run": "crop_2026-02-10_10-00-00",
                "review_state": None,
                "review_intended_use": None,
            },
            {
                "dataset_id": "dataset_b",
                "zarr_path": zarr_b,
                "recording_id": "recording_b",
                "crop_run": "crop_2026-02-10_10-00-00",
                "review_state": None,
                "review_intended_use": None,
            },
        ],
    )

    output = tmp_path / "crop_review_list_registry.txt"
    rc = mod.main(
        [
            str(rec_root),
            "--recursive",
            "--stage",
            "crop",
            "--registry",
            str(registry_path),
            "--output",
            str(output),
        ]
    )
    assert rc == 0

    lines = [line.strip() for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines == [str(zarr_a.resolve())]


def test_generate_review_list_crop_registry_mode_any_status_includes_running(tmp_path: Path) -> None:
    rec_root = tmp_path / "recordings"
    rec_root.mkdir()
    zarr_a = rec_root / "a_analysis.zarr"
    zarr_b = rec_root / "b_analysis.zarr"
    _make_crop_zarr(zarr_a, run_status="completed", review_state=None)
    _make_crop_zarr(zarr_b, run_status="running", review_state=None)

    registry_path = tmp_path / "registry.sqlite"
    _seed_crop_registry(
        registry_path,
        rows=[
            {
                "dataset_id": "dataset_a",
                "zarr_path": zarr_a,
                "recording_id": "recording_a",
                "crop_run": "crop_2026-02-10_10-00-00",
                "review_state": None,
                "review_intended_use": None,
            },
            {
                "dataset_id": "dataset_b",
                "zarr_path": zarr_b,
                "recording_id": "recording_b",
                "crop_run": "crop_2026-02-10_10-00-00",
                "review_state": None,
                "review_intended_use": None,
            },
        ],
    )

    output = tmp_path / "crop_review_list_registry.txt"
    rc = mod.main(
        [
            str(rec_root),
            "--recursive",
            "--stage",
            "crop",
            "--registry",
            str(registry_path),
            "--crop-run-status",
            "any",
            "--output",
            str(output),
        ]
    )
    assert rc == 0

    lines = [line.strip() for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines == sorted([str(zarr_a.resolve()), str(zarr_b.resolve())])


def test_generate_review_list_crop_registry_mode_treats_stale_reviews_as_missing(tmp_path: Path) -> None:
    rec_root = tmp_path / "recordings"
    rec_root.mkdir()
    zarr_fresh = rec_root / "fresh_analysis.zarr"
    zarr_stale = rec_root / "stale_analysis.zarr"
    _make_crop_zarr(zarr_fresh, run_status="completed", review_state="approved")
    _make_crop_zarr(zarr_stale, run_status="completed", review_state="approved")

    registry_path = tmp_path / "registry.sqlite"
    _seed_crop_registry(
        registry_path,
        rows=[
            {
                "dataset_id": "dataset_fresh",
                "zarr_path": zarr_fresh,
                "recording_id": "recording_fresh",
                "crop_run": "crop_2026-02-10_10-00-00",
                "review_state": "approved",
                "review_intended_use": "training",
            },
            {
                "dataset_id": "dataset_stale",
                "zarr_path": zarr_stale,
                "recording_id": "recording_stale",
                "crop_run": "crop_2026-02-10_10-00-00",
                "review_state": "approved",
                "review_intended_use": "training",
                "zarr_mtime_ns": int(zarr_stale.stat().st_mtime_ns + 1),
            },
        ],
    )

    output = tmp_path / "crop_review_list_registry_stale.txt"
    rc = mod.main(
        [
            str(rec_root),
            "--recursive",
            "--stage",
            "crop",
            "--registry",
            str(registry_path),
            "--review-state",
            "approved",
            "--review-method",
            "manual",
            "--review-intended-use",
            "training",
            "--output",
            str(output),
        ]
    )
    assert rc == 0

    lines = [line.strip() for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines == [str(zarr_fresh.resolve())]
