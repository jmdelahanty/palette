from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import zarr

from fisheye.registry.db import Registry
from fisheye.utils import check_recording_steps as mod


def _write_minimal_h5(path: Path, *, session_uuid: str, camera_id: str = "cam_1") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.attrs["session_uuid"] = session_uuid
        h5.attrs["camera_id"] = camera_id


def test_check_zarr_reports_detect_coverage_full_basis(tmp_path: Path) -> None:
    zarr_path = tmp_path / "full_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    raw = root.create_group("raw_video")
    raw.create_array("images_full", data=np.zeros((4,), dtype=np.uint8))
    detect_parent = root.create_group("detect_runs")
    detect_parent.attrs["latest"] = "detect_full"
    run = detect_parent.create_group("detect_full")
    run.create_array("frame_counts", data=np.array([1, 0, 2, 1], dtype=np.int32))

    info = mod._check_zarr(zarr_path, tuning_keys=[])  # noqa: SLF001

    assert info["detect_present"] is True
    assert info["detect_method"] is None
    assert info["detect_coverage_basis"] == "full"
    assert info["detect_coverage"] == pytest.approx(75.0)


def test_check_zarr_reports_detect_coverage_sampled_basis(tmp_path: Path) -> None:
    zarr_path = tmp_path / "sampled_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    raw = root.create_group("raw_video")
    raw.create_array("original_frame_indices", data=np.array([0, 2, 4], dtype=np.int32))
    detect_parent = root.create_group("detect_runs")
    detect_parent.attrs["latest"] = "detect_sampled"
    run = detect_parent.create_group("detect_sampled")
    run.create_array("frame_counts", data=np.array([1, 0, 1], dtype=np.int32))

    info = mod._check_zarr(zarr_path, tuning_keys=[])  # noqa: SLF001

    assert info["detect_present"] is True
    assert info["detect_coverage_basis"] == "sampled"
    assert info["detect_coverage"] == pytest.approx(66.6667, abs=1e-3)


def test_check_zarr_reports_detect_coverage_inferred_basis(tmp_path: Path) -> None:
    zarr_path = tmp_path / "inferred_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    detect_parent = root.create_group("detect_runs")
    detect_parent.attrs["latest"] = "detect_inferred"
    run = detect_parent.create_group("detect_inferred")
    run.create_array("frame_indices", data=np.array([0, 0, 2, 5], dtype=np.int32))

    info = mod._check_zarr(zarr_path, tuning_keys=[])  # noqa: SLF001

    assert info["detect_present"] is True
    assert info["detect_coverage_basis"] == "inferred"
    assert info["detect_coverage"] == pytest.approx(50.0)


def test_check_zarr_reports_detect_missing_when_no_detect_runs(tmp_path: Path) -> None:
    zarr_path = tmp_path / "empty_analysis.zarr"
    zarr.open_group(str(zarr_path), mode="w")

    info = mod._check_zarr(zarr_path, tuning_keys=[])  # noqa: SLF001

    assert info["detect_present"] is False
    assert info["detect_method"] is None
    assert info["detect_coverage"] is None
    assert info["detect_coverage_basis"] is None
    assert info["detect_quality_present"] is False
    assert info["detect_quality_run"] is None


def test_check_zarr_reads_detect_method_from_provenance(tmp_path: Path) -> None:
    zarr_path = tmp_path / "method_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    detect_parent = root.create_group("detect_runs")
    detect_parent.attrs["latest"] = "detect_prov"
    run = detect_parent.create_group("detect_prov")
    run.attrs["provenance"] = {"method": "yolo"}
    run.create_array("frame_counts", data=np.array([1, 1], dtype=np.int32))

    info = mod._check_zarr(zarr_path, tuning_keys=[])  # noqa: SLF001

    assert info["detect_present"] is True
    assert info["detect_method"] == "yolo"


def test_check_zarr_reads_detect_quality_summary(tmp_path: Path) -> None:
    zarr_path = tmp_path / "quality_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    detect_parent = root.create_group("detect_runs")
    detect_parent.attrs["latest"] = "detect_quality_source"
    run = detect_parent.create_group("detect_quality_source")
    run.create_array("frame_counts", data=np.array([1, 1, 0], dtype=np.int32))
    quality_parent = run.create_group("quality_reports")
    quality_parent.attrs["latest"] = "detect_quality_2026-02-09_12-00-00"
    quality = quality_parent.create_group("detect_quality_2026-02-09_12-00-00")
    quality.attrs["quality_score"] = {"grade": "A", "overall_score": 99.2}
    quality.attrs["detection_quality_summary"] = {
        "clean_percentage": 98.7,
        "blip_detections": 3,
        "jump_detections": 2,
        "multi_detections": 1,
    }

    info = mod._check_zarr(zarr_path, tuning_keys=[])  # noqa: SLF001

    assert info["detect_quality_present"] is True
    assert info["detect_quality_run"] == "detect_quality_2026-02-09_12-00-00"
    assert info["detect_quality_grade"] == "A"
    assert info["detect_quality_score"] == pytest.approx(99.2)
    assert info["detect_quality_clean_percent"] == pytest.approx(98.7)
    assert info["detect_quality_artifacts"] == 6


def test_check_zarr_reads_crop_status_from_latest_run(tmp_path: Path) -> None:
    zarr_path = tmp_path / "crop_status_latest_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_2026-02-10_12-00-00"
    crop_run = crop_parent.create_group("crop_2026-02-10_12-00-00")
    crop_run.attrs["status"] = "completed"

    info = mod._check_zarr(zarr_path, tuning_keys=[])  # noqa: SLF001

    assert info["crop_present"] is True
    assert info["crop_status"] == "completed"


def test_check_zarr_reads_crop_status_from_fallback_latest_name(tmp_path: Path) -> None:
    zarr_path = tmp_path / "crop_status_fallback_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    crop_parent = root.create_group("crop_runs")
    crop_parent.create_group("crop_2026-02-10_11-00-00").attrs["status"] = "failed"
    crop_parent.create_group("crop_2026-02-10_12-00-00").attrs["status"] = "running"

    info = mod._check_zarr(zarr_path, tuning_keys=[])  # noqa: SLF001

    assert info["crop_present"] is True
    assert info["crop_status"] == "running"


def test_crop_status_format_helpers() -> None:
    assert mod._crop_status_text(False, None) == "MISS"  # noqa: SLF001
    assert mod._crop_status_text(True, None) == "OK"  # noqa: SLF001
    assert mod._crop_status_text(True, "FAILED") == "failed"  # noqa: SLF001
    assert mod._crop_status_rich(True, "completed") == "[chartreuse1]completed[/chartreuse1]"  # noqa: SLF001
    assert mod._crop_status_rich(True, "running") == "[yellow]running[/yellow]"  # noqa: SLF001
    assert mod._crop_status_rich(True, "failed") == "[red]failed[/red]"  # noqa: SLF001


def test_registry_crop_review_status_for_zarr_returns_latest_review_fields(tmp_path: Path) -> None:
    zarr_path = tmp_path / "a_analysis.zarr"
    zarr.open_group(str(zarr_path), mode="w")

    registry_path = tmp_path / "registry.sqlite"
    registry = Registry(registry_path)
    registry.upsert_dataset(
        "dataset_a",
        session_uuid="session_a",
        zarr_path=zarr_path,
        recording_id="recording_a",
        artifact_kind="source_recording",
        zarr_use="analysis",
    )
    registry.upsert_provenance(
        "dataset_a",
        provenance={},
        context={},
        protocol_name=None,
        protocol_hash=None,
        acquisition={},
        zarr_purpose="analysis",
    )
    registry.replace_crop_quality(
        "dataset_a",
        [
            {
                "crop_run": "crop_2026-02-10_10-00-00",
                "recording_id": "recording_a",
                "zarr_use": "analysis",
                "crop_created_utc": "2026-02-10T10:00:00+00:00",
                "source_detect_run": None,
                "source_refined_run": None,
                "detection_source_type": "manual",
                "detection_source_path": None,
                "total_rois": 100,
                "frames_with_crops": 90,
                "total_frames": 100,
                "percent_frames_with_crops": 90.0,
                "includes_interpolated": 0,
                "n_real_detections": 100,
                "n_interpolated_detections": 0,
                "review_state": "approved",
                "review_method": "manual",
                "review_intended_use": "training",
                "review_reviewer": "alice",
                "review_timestamp_utc": "2026-02-10T10:30:00+00:00",
                "review_notes": "looks good",
                "zarr_mtime_ns": 12345,
                "updated_utc": "2026-02-10T10:30:00+00:00",
            }
        ],
    )

    payload = mod._registry_crop_review_status_for_zarr(  # noqa: SLF001
        registry=registry,
        zarr_path=zarr_path,
    )
    registry.close()

    assert payload is not None
    assert payload["crop_run"] == "crop_2026-02-10_10-00-00"
    status = payload["crop_review_status"]
    assert isinstance(status, dict)
    assert status["state"] == "approved"
    assert status["method"] == "manual"
    assert status["intended_use"] == "training"
    assert status["reviewer"] == "alice"
    assert status["timestamp_utc"] == "2026-02-10T10:30:00+00:00"


def test_open_root_live_prefers_non_consolidated(monkeypatch) -> None:
    sentinel = object()
    calls: list[dict[str, object]] = []

    def _fake_open_group(path: str, **kwargs):  # type: ignore[no-untyped-def]
        calls.append({"path": path, **kwargs})
        return sentinel

    monkeypatch.setattr(mod.zarr, "open_group", _fake_open_group)

    out = mod._open_root_live(Path("/tmp/example.zarr"))  # noqa: SLF001

    assert out is sentinel
    assert len(calls) == 1
    assert calls[0]["path"] == "/tmp/example.zarr"
    assert calls[0]["mode"] == "r"
    assert calls[0]["use_consolidated"] is False


def test_open_root_live_falls_back_when_use_consolidated_unsupported(monkeypatch) -> None:
    sentinel = object()
    calls: list[dict[str, object]] = []

    def _fake_open_group(path: str, **kwargs):  # type: ignore[no-untyped-def]
        calls.append({"path": path, **kwargs})
        if "use_consolidated" in kwargs:
            raise TypeError("unsupported kwarg")
        return sentinel

    monkeypatch.setattr(mod.zarr, "open_group", _fake_open_group)

    out = mod._open_root_live(Path("/tmp/example.zarr"))  # noqa: SLF001

    assert out is sentinel
    assert len(calls) == 2
    assert calls[0]["mode"] == "r"
    assert calls[0]["use_consolidated"] is False
    assert calls[1]["mode"] == "r"
    assert "use_consolidated" not in calls[1]


def test_check_zarr_reads_refined_eye_mask_review_status_from_latest_run(tmp_path: Path) -> None:
    zarr_path = tmp_path / "eye_review_latest_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    refined_parent = root.create_group("refined_eye_masks_runs")
    refined_parent.attrs["latest"] = "refined_eye_masks_001"
    run = refined_parent.create_group("refined_eye_masks_001")
    run.attrs["eye_mask_review_status"] = {
        "state": "approved",
        "method": "manual",
        "intended_use": "training",
        "reviewer": "alice",
        "timestamp_utc": "2026-02-12T04:50:00+00:00",
    }

    info = mod._check_zarr(zarr_path, tuning_keys=[])  # noqa: SLF001

    assert info["refined_eye_masks_present"] is True
    status = info["eye_mask_review_status"]
    assert isinstance(status, dict)
    assert status["state"] == "approved"
    assert status["method"] == "manual"
    assert status["intended_use"] == "training"


def test_check_zarr_reads_refined_eye_mask_review_status_from_parent_latest_pointer(tmp_path: Path) -> None:
    zarr_path = tmp_path / "eye_review_pointer_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    refined_parent = root.create_group("refined_eye_masks_runs")
    refined_parent.attrs["latest"] = "refined_eye_masks_002"
    refined_parent.attrs["eye_mask_review_status_latest"] = "refined_eye_masks_001"
    refined_parent.create_group("refined_eye_masks_002")
    run = refined_parent.create_group("refined_eye_masks_001")
    run.attrs["eye_mask_review_status"] = {
        "state": "pending",
        "method": "spotcheck",
        "intended_use": "training",
    }

    info = mod._check_zarr(zarr_path, tuning_keys=[])  # noqa: SLF001

    assert info["refined_eye_masks_present"] is True
    status = info["eye_mask_review_status"]
    assert isinstance(status, dict)
    assert status["state"] == "pending"
    assert status["method"] == "spotcheck"


def test_main_status_source_registry_requires_registry(tmp_path: Path, capsys) -> None:
    root = tmp_path / "recordings"
    root.mkdir(parents=True)

    rc = mod.main([str(root), "--status-source", "registry", "--no-rich"])
    out = capsys.readouterr().out

    assert rc == 1
    assert "--status-source registry requires --registry." in out


def test_main_status_source_registry_uses_registry_views(tmp_path: Path, monkeypatch, capsys) -> None:
    recordings_root = tmp_path / "recordings"
    h5_path = recordings_root / "rec_registry" / "raw" / "capture.h5"
    _write_minimal_h5(h5_path, session_uuid="rec_registry", camera_id="cam_9")

    zarr_path = recordings_root / "rec_registry" / "zarr" / "rec_registry_analysis.zarr"

    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset(
        "dataset_registry_a",
        session_uuid="rec_registry",
        zarr_path=zarr_path,
        recording_id="rec_registry",
        artifact_kind="source_recording",
        zarr_use="analysis",
    )
    registry.conn.execute(
        """
        INSERT INTO recording_step_status (
            dataset_id, recording_id, step_name, status, run_name, method, coverage_pct, source, updated_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            "dataset_registry_a",
            "rec_registry",
            "detect",
            "ok",
            "detect_001",
            "yolo",
            99.0,
            "unit_test",
            "2026-02-22T00:00:00+00:00",
        ),
    )
    registry.conn.commit()
    registry.close()

    def _fail_check_zarr(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("filesystem zarr traversal should not run in registry mode")

    monkeypatch.setattr(mod, "_check_zarr", _fail_check_zarr)

    rc = mod.main(
        [
            str(recordings_root),
            "--status-source",
            "registry",
            "--registry",
            str(tmp_path / "registry.sqlite"),
            "--no-rich",
        ]
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert "rec_registry" in out
    assert "detect: OK" in out


def test_main_status_source_compare_outputs_mismatches(tmp_path: Path, capsys) -> None:
    recordings_root = tmp_path / "recordings"
    h5_path = recordings_root / "rec_compare" / "raw" / "capture.h5"
    _write_minimal_h5(h5_path, session_uuid="rec_compare", camera_id="cam_2")

    zarr_path = recordings_root / "rec_compare" / "zarr" / "rec_compare_analysis.zarr"

    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset(
        "dataset_compare_a",
        session_uuid="rec_compare",
        zarr_path=zarr_path,
        recording_id="rec_compare",
        artifact_kind="source_recording",
        zarr_use="analysis",
    )
    registry.conn.execute(
        """
        INSERT INTO recording_step_status (
            dataset_id, recording_id, step_name, status, run_name, source, updated_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """,
        (
            "dataset_compare_a",
            "rec_compare",
            "detect",
            "ok",
            "detect_compare",
            "unit_test",
            "2026-02-22T00:00:00+00:00",
        ),
    )
    registry.conn.commit()
    registry.close()

    rc = mod.main(
        [
            str(recordings_root),
            "--status-source",
            "compare",
            "--registry",
            str(tmp_path / "registry.sqlite"),
            "--no-rich",
        ]
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert "Recording Step Status Mismatches" in out
    assert "field: detect" in out


def test_registry_status_payload_filters_rows_to_matching_recording_id(tmp_path: Path) -> None:
    zarr_path = tmp_path / "shared_training.zarr"
    zarr.open_group(str(zarr_path), mode="w")

    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset(
        "dataset_good",
        session_uuid="session_good",
        zarr_path=zarr_path,
        recording_id="rec_target",
        artifact_kind="source_recording",
        zarr_use="training",
    )
    registry.upsert_dataset(
        "dataset_ghost",
        session_uuid="session_ghost",
        zarr_path=zarr_path,
        recording_id=None,
        artifact_kind="source_recording",
        zarr_use="training",
    )
    registry.conn.executemany(
        """
        INSERT INTO recording_step_status (
            dataset_id, recording_id, step_name, status, source, updated_utc
        )
        VALUES (?, ?, ?, ?, ?, ?);
        """,
        [
            ("dataset_good", "rec_target", "eye_masks", "ok", "unit_test", "2026-02-23T01:00:00+00:00"),
            ("dataset_good", "rec_target", "refined_eye_masks", "ok", "unit_test", "2026-02-23T01:00:00+00:00"),
            ("dataset_good", "rec_target", "eye_mask_tuning", "ok", "unit_test", "2026-02-23T01:00:00+00:00"),
            ("dataset_ghost", None, "eye_masks", "missing", "unit_test", "2026-02-23T02:00:00+00:00"),
            ("dataset_ghost", None, "refined_eye_masks", "missing", "unit_test", "2026-02-23T02:00:00+00:00"),
            ("dataset_ghost", None, "eye_mask_tuning", "missing", "unit_test", "2026-02-23T02:00:00+00:00"),
        ],
    )
    registry.conn.commit()

    payload = mod._registry_status_payload(  # noqa: SLF001
        registry=registry,
        zarr_path=zarr_path,
        recording_id="rec_target",
        tuning_keys=["eye_mask_tuning"],
    )
    registry.close()

    assert payload["eye_masks_present"] is True
    assert payload["refined_eye_masks_present"] is True
    assert payload["tuning_status"]["eye_mask_tuning"] == "ok"  # type: ignore[index]
