"""Tests for registry_query subject-lineage filters."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.utils.registry_query import main as registry_query_main


def _seed_registry_for_subject_filters(registry_path: Path) -> None:
    registry = Registry(registry_path)
    # Minimal dataset rows.
    registry.upsert_dataset(
        "dataset_a",
        session_uuid="session_a",
        zarr_path=registry_path.parent / "a.zarr",
        recording_id="recording_a",
        artifact_kind="source_recording",
    )
    registry.upsert_dataset(
        "dataset_b",
        session_uuid="session_b",
        zarr_path=registry_path.parent / "b.zarr",
        recording_id="recording_b",
        artifact_kind="source_recording",
    )
    registry.upsert_provenance(
        "dataset_a",
        provenance={},
        context={},
        protocol_name=None,
        protocol_hash=None,
        acquisition={},
        zarr_purpose=None,
    )
    registry.upsert_provenance(
        "dataset_b",
        provenance={},
        context={},
        protocol_name=None,
        protocol_hash=None,
        acquisition={},
        zarr_purpose=None,
    )
    # Recording context rows for view joins.
    registry.conn.execute(
        """
        INSERT INTO recordings (
            recording_id, session_uuid, recording_name, recording_path, recording_type,
            recording_subtype, behavior_mode, artifact_schema_id, created_utc, updated_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'));
        """,
        (
            "recording_a",
            "session_a",
            "recording_a",
            str(registry_path.parent / "recording_a"),
            "behavior",
            "free",
            "free",
            "behavior_v1",
        ),
    )
    registry.conn.execute(
        """
        INSERT INTO recordings (
            recording_id, session_uuid, recording_name, recording_path, recording_type,
            recording_subtype, behavior_mode, artifact_schema_id, created_utc, updated_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'));
        """,
        (
            "recording_b",
            "session_b",
            "recording_b",
            str(registry_path.parent / "recording_b"),
            "behavior",
            "free",
            "free",
            "behavior_v1",
        ),
    )
    # Lineage entities.
    registry.conn.execute(
        """
        INSERT INTO crosses (cross_id, genotype, created_utc, updated_utc)
        VALUES ('cross_a', 'genotype_x', datetime('now'), datetime('now'));
        """
    )
    registry.conn.execute(
        """
        INSERT INTO crosses (cross_id, genotype, created_utc, updated_utc)
        VALUES ('cross_b', 'genotype_y', datetime('now'), datetime('now'));
        """
    )
    registry.conn.execute(
        """
        INSERT INTO dishes (dish_id, cross_id, created_utc, updated_utc)
        VALUES ('dish_a', 'cross_a', datetime('now'), datetime('now'));
        """
    )
    registry.conn.execute(
        """
        INSERT INTO dishes (dish_id, cross_id, created_utc, updated_utc)
        VALUES ('dish_b', 'cross_b', datetime('now'), datetime('now'));
        """
    )
    registry.conn.execute(
        """
        INSERT INTO subjects (subject_id, dish_id, created_utc, updated_utc)
        VALUES ('subject_a', 'dish_a', datetime('now'), datetime('now'));
        """
    )
    registry.conn.execute(
        """
        INSERT INTO subjects (subject_id, dish_id, created_utc, updated_utc)
        VALUES ('subject_b', 'dish_b', datetime('now'), datetime('now'));
        """
    )
    registry.conn.execute(
        """
        INSERT INTO recording_subjects (
            recording_id, subject_id, dataset_id, dish_id, cross_id, dpf_at_acquisition, created_utc, updated_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'));
        """,
        ("recording_a", "subject_a", "dataset_a", "dish_a", "cross_a", 8),
    )
    registry.conn.execute(
        """
        INSERT INTO recording_subjects (
            recording_id, subject_id, dataset_id, dish_id, cross_id, dpf_at_acquisition, created_utc, updated_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'));
        """,
        ("recording_b", "subject_b", "dataset_b", "dish_b", "cross_b", 12),
    )
    registry.conn.commit()
    registry.close()


def _seed_registry_for_detect_filters(registry_path: Path) -> None:
    registry = Registry(registry_path)
    for (
        dataset_id,
        session_uuid,
        recording_id,
        filename,
        rig_id,
        arena_id,
        camera_id,
        dish_design,
    ) in (
        ("dataset_a", "session_a", "recording_a", "a.zarr", "rig_a", "arena_x", "cam_1", "cedar"),
        ("dataset_b", "session_b", "recording_b", "b.zarr", "rig_a", "arena_x", "cam_2", "cedar"),
        ("dataset_c", "session_c", "recording_c", "c.zarr", "rig_b", "arena_y", "cam_3", "maple"),
    ):
        registry.upsert_dataset(
            dataset_id,
            session_uuid=session_uuid,
            zarr_path=registry_path.parent / filename,
            recording_id=recording_id,
            artifact_kind="source_recording",
            zarr_use="analysis",
        )
        registry.upsert_provenance(
            dataset_id,
            provenance={},
            context={
                "rig_id": rig_id,
                "arena_id": arena_id,
                "camera_id": camera_id,
            },
            protocol_name=None,
            protocol_hash=None,
            acquisition={
                "dish_design": dish_design,
            },
            zarr_purpose="analysis",
        )

    registry.upsert_detect_performance(
        dataset_id="dataset_a",
        detect_run="detect_a",
        detect_created_utc="2026-02-09T00:00:00+00:00",
        recording_id="recording_a",
        zarr_use="analysis",
        detection_method="yolo",
        model_run_id="run_detect_model_v1",
        model_set_id="detect_set_v1",
        model_path="/models/detect_model_v1.pt",
        model_name="detect_model_v1.pt",
        coverage_percent=95.0,
        frames_with_detections=95,
        frames_zero_detections=5,
        total_frames=100,
        mean_confidence=0.9,
        min_confidence=0.5,
        max_confidence=1.0,
        inference_duration_seconds=10.0,
        inference_average_fps=120.0,
        inference_avg_batch_ms=50.0,
        inference_avg_read_ms=80.0,
        conf_threshold=0.4,
        iou_threshold=0.8,
        batch_size=16,
        inference_width=640,
        inference_height=640,
    )
    registry.upsert_detect_performance(
        dataset_id="dataset_b",
        detect_run="detect_b",
        detect_created_utc="2026-02-09T00:00:00+00:00",
        recording_id="recording_b",
        zarr_use="analysis",
        detection_method="traditional",
        model_run_id=None,
        model_set_id=None,
        model_path=None,
        model_name=None,
        coverage_percent=70.0,
        frames_with_detections=70,
        frames_zero_detections=30,
        total_frames=100,
        mean_confidence=0.8,
        min_confidence=0.4,
        max_confidence=1.0,
        inference_duration_seconds=20.0,
        inference_average_fps=60.0,
        inference_avg_batch_ms=80.0,
        inference_avg_read_ms=130.0,
        conf_threshold=None,
        iou_threshold=None,
        batch_size=16,
        inference_width=640,
        inference_height=640,
    )
    registry.upsert_detect_performance(
        dataset_id="dataset_c",
        detect_run="detect_c",
        detect_created_utc="2026-02-09T00:00:00+00:00",
        recording_id="recording_c",
        zarr_use="analysis",
        detection_method="yolo",
        model_run_id="run_detect_model_v2",
        model_set_id="detect_set_v2",
        model_path="/models/detect_model_v2.pt",
        model_name="detect_model_v2.pt",
        coverage_percent=85.0,
        frames_with_detections=85,
        frames_zero_detections=15,
        total_frames=100,
        mean_confidence=0.88,
        min_confidence=0.45,
        max_confidence=1.0,
        inference_duration_seconds=12.0,
        inference_average_fps=100.0,
        inference_avg_batch_ms=55.0,
        inference_avg_read_ms=90.0,
        conf_threshold=0.4,
        iou_threshold=0.8,
        batch_size=16,
        inference_width=640,
        inference_height=640,
    )

    def _crop_record(
        *,
        crop_run: str,
        recording_id: str,
        zarr_use: str,
        source_type: str,
        percent_frames: float,
        review_state: str | None,
        review_intended_use: str | None,
        review_method: str | None = "manual",
    ) -> dict[str, object]:
        return {
            "crop_run": crop_run,
            "recording_id": recording_id,
            "zarr_use": zarr_use,
            "crop_created_utc": "2026-02-09T00:00:00+00:00",
            "source_detect_run": None,
            "source_refined_run": "refined_detect_2026-02-09_00-00-00",
            "detection_source_type": source_type,
            "detection_source_path": "refined_detect_runs/refined_detect_2026-02-09_00-00-00/manual",
            "total_rois": 1000,
            "frames_with_crops": int(percent_frames),
            "total_frames": 100,
            "percent_frames_with_crops": percent_frames,
            "includes_interpolated": 0,
            "n_real_detections": 1000,
            "n_interpolated_detections": 0,
            "review_state": review_state,
            "review_method": review_method,
            "review_intended_use": review_intended_use,
            "review_reviewer": "tester",
            "review_timestamp_utc": "2026-02-09T00:05:00+00:00",
            "review_notes": None,
            "zarr_mtime_ns": 123456789,
            "updated_utc": "2026-02-09T00:05:00+00:00",
        }

    registry.replace_crop_quality(
        "dataset_a",
        [
            _crop_record(
                crop_run="crop_a",
                recording_id="recording_a",
                zarr_use="analysis",
                source_type="full_recording",
                percent_frames=95.0,
                review_state="approved",
                review_intended_use="training",
            )
        ],
    )
    registry.replace_crop_quality(
        "dataset_b",
        [
            _crop_record(
                crop_run="crop_b",
                recording_id="recording_b",
                zarr_use="analysis",
                source_type="raw",
                percent_frames=70.0,
                review_state=None,
                review_intended_use=None,
            )
        ],
    )
    registry.replace_crop_quality(
        "dataset_c",
        [
            _crop_record(
                crop_run="crop_c",
                recording_id="recording_c",
                zarr_use="analysis",
                source_type="interpolated",
                percent_frames=85.0,
                review_state="needs_review",
                review_intended_use="full_recording",
            )
        ],
    )
    registry.close()


def test_registry_query_filters_by_cross_id(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_subject_filters(registry_path)

    rc = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--cross-id",
            "cross_a",
            "--json",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    dataset_ids = {row["dataset_id"] for row in payload}
    assert dataset_ids == {"dataset_a"}


def test_registry_query_filters_by_genotype_and_dpf(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_subject_filters(registry_path)

    rc = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--genotype",
            "genotype_y",
            "--dpf",
            "12",
            "--json",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    dataset_ids = {row["dataset_id"] for row in payload}
    assert dataset_ids == {"dataset_b"}


def test_registry_query_filters_by_dpf_range(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_subject_filters(registry_path)

    rc = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--dpf-min",
            "9",
            "--dpf-max",
            "12",
            "--json",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    dataset_ids = {row["dataset_id"] for row in payload}
    assert dataset_ids == {"dataset_b"}


def test_registry_query_rejects_invalid_dpf_range(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_subject_filters(registry_path)

    try:
        registry_query_main(
            [
                "--registry",
                str(registry_path),
                "--dpf-min",
                "13",
                "--dpf-max",
                "12",
                "--json",
            ]
        )
    except SystemExit as exc:
        assert "--dpf-min must be <= --dpf-max." in str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("Expected SystemExit for invalid DPF range.")


def test_registry_query_filters_by_detect_coverage_min(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)

    rc = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--detect-coverage-min",
            "90",
            "--json",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    dataset_ids = {row["dataset_id"] for row in payload}
    assert dataset_ids == {"dataset_a"}


def test_registry_query_detect_model_only_and_model_like(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)

    rc = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--detect-model-only",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    dataset_ids = {row["dataset_id"] for row in payload}
    assert dataset_ids == {"dataset_a", "dataset_c"}
    runs = {row.get("detect_model_run_id") for row in payload}
    assert runs == {"run_detect_model_v1", "run_detect_model_v2"}

    rc2 = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--detect-model-like",
            "v2",
            "--json",
        ]
    )
    assert rc2 == 0
    payload2 = json.loads(capsys.readouterr().out)
    dataset_ids2 = {row["dataset_id"] for row in payload2}
    assert dataset_ids2 == {"dataset_c"}


def test_registry_query_group_by_model_json(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)

    rc = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--group-by-model",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 2
    names = {row["model_name"] for row in payload}
    assert names == {"detect_model_v1.pt", "detect_model_v2.pt"}
    run_ids = {row["model_run_id"] for row in payload}
    assert run_ids == {"run_detect_model_v1", "run_detect_model_v2"}
    counts = {row["model_name"]: row["recordings"] for row in payload}
    assert counts["detect_model_v1.pt"] == 1
    assert counts["detect_model_v2.pt"] == 1


def test_registry_query_group_by_dimension_json_includes_percentiles(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)

    rc = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--group-by",
            "rig",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 2
    by_group = {row["group_value"]: row for row in payload}
    assert set(by_group.keys()) == {"rig_a", "rig_b"}

    rig_a = by_group["rig_a"]
    assert rig_a["datasets"] == 2
    assert rig_a["recordings"] == 2
    assert rig_a["coverage_avg"] == 82.5
    assert rig_a["coverage_p50"] == 82.5
    assert rig_a["fps_p50"] == 90.0
    assert rig_a["read_ms_p50"] == 105.0

    for key in (
        "coverage_p10",
        "coverage_p50",
        "coverage_p90",
        "fps_p10",
        "fps_p50",
        "fps_p90",
        "read_ms_p10",
        "read_ms_p50",
        "read_ms_p90",
    ):
        assert key in rig_a


def test_registry_query_group_by_model_alias_matches_explicit(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)

    rc_alias = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--group-by-model",
            "--json",
        ]
    )
    assert rc_alias == 0
    alias_payload = json.loads(capsys.readouterr().out)

    rc_explicit = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--group-by",
            "model",
            "--json",
        ]
    )
    assert rc_explicit == 0
    explicit_payload = json.loads(capsys.readouterr().out)
    assert alias_payload == explicit_payload


def test_registry_query_rejects_group_by_model_conflict(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)

    try:
        registry_query_main(
            [
                "--registry",
                str(registry_path),
                "--group-by-model",
                "--group-by",
                "rig",
            ]
        )
    except SystemExit as exc:
        assert "--group-by-model cannot be combined with --group-by non-model values." in str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("Expected SystemExit for conflicting group-by args.")


def test_registry_query_detect_model_summary_mode_json(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)

    rc = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--detect-model-summary",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 2
    run_ids = {row["model_run_id"] for row in payload}
    assert run_ids == {"run_detect_model_v1", "run_detect_model_v2"}
    for row in payload:
        assert "coverage_p50" in row
        assert "fps_p50" in row
        assert "read_ms_p50" in row


def test_registry_query_detect_model_summary_mode_filters(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)

    rc = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--detect-model-summary",
            "--detect-model-like",
            "v2",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 1
    assert payload[0]["model_run_id"] == "run_detect_model_v2"


def test_registry_query_detect_model_summary_rejects_output_file_list(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)
    out_file = tmp_path / "rows.txt"

    try:
        registry_query_main(
            [
                "--registry",
                str(registry_path),
                "--detect-model-summary",
                "--output-file-list",
                str(out_file),
            ]
        )
    except SystemExit as exc:
        assert "--output-file-list is only supported for dataset-row query mode." in str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("Expected SystemExit for invalid output-file-list usage.")


def test_registry_query_filters_by_crop_review_state(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)

    rc = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--crop-review-state",
            "approved",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    dataset_ids = {row["dataset_id"] for row in payload}
    assert dataset_ids == {"dataset_a"}
    assert payload[0]["crop_review_state"] == "approved"


def test_registry_query_filters_by_crop_missing_review_state(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)

    rc = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--crop-review-state",
            "missing",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    dataset_ids = {row["dataset_id"] for row in payload}
    assert dataset_ids == {"dataset_b"}
    assert payload[0]["crop_review_state"] is None


def test_registry_query_filters_by_crop_source_and_intended_use(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)

    rc = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--crop-review-intended-use",
            "full_recording",
            "--crop-source-type",
            "interpolated",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    dataset_ids = {row["dataset_id"] for row in payload}
    assert dataset_ids == {"dataset_c"}
    assert payload[0]["crop_source_type"] == "interpolated"


def test_registry_query_filters_by_crop_percent_frames_threshold(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)

    rc = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--crop-percent-frames-min",
            "90",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    dataset_ids = {row["dataset_id"] for row in payload}
    assert dataset_ids == {"dataset_a"}
    assert payload[0]["crop_percent_frames_with_crops"] == 95.0
