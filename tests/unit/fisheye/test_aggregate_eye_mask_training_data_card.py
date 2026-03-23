from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.utils import aggregate_eye_mask_training_data_card as mod


def _seed_dataset(registry: Registry, *, dataset_id: str, recording_id: str, zarr_path: Path) -> None:
    zarr_path.mkdir(parents=True, exist_ok=True)
    registry.upsert_dataset(
        dataset_id,
        session_uuid=f"{dataset_id}_session",
        zarr_path=zarr_path,
        recording_id=recording_id,
        artifact_kind="source_recording",
        zarr_use="training",
    )


def _seed_legacy_provenance_context(
    registry: Registry,
    *,
    dataset_id: str,
    rig_id: str,
    arena_id: str,
    camera_id: str,
    canvas_name: str,
    protocol_name: str,
    dish_design: str,
) -> None:
    registry.upsert_provenance(
        dataset_id,
        provenance={"snapshot_status": "complete"},
        context={
            "rig_id": rig_id,
            "arena_id": arena_id,
            "camera_id": camera_id,
            "canvas_name": canvas_name,
        },
        protocol_name=protocol_name,
        protocol_hash="hash_ctx",
        acquisition={"dish_design": dish_design},
        zarr_purpose="analysis",
    )


def _seed_recording_context(
    registry: Registry,
    *,
    recording_id: str,
    session_uuid: str,
    root: Path,
    rig_id: str,
    arena_id: str,
    camera_id: str,
    canvas_name: str,
    protocol_name: str,
    dish_design: str,
) -> None:
    registry.conn.execute(
        """
        INSERT INTO recordings (
            recording_id, session_uuid, recording_name, recording_path,
            recording_type, recording_subtype, behavior_mode, artifact_schema_id,
            rig_id, arena_id, camera_id, canvas_name, protocol_name, dish_design,
            created_utc, updated_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'));
        """,
        (
            recording_id,
            session_uuid,
            recording_id,
            str(root / "recordings" / recording_id),
            "behavior",
            "free",
            "free",
            "behavior_v1",
            rig_id,
            arena_id,
            camera_id,
            canvas_name,
            protocol_name,
            dish_design,
        ),
    )
    registry.conn.commit()


def _seed_subject_lineage(
    registry: Registry,
    *,
    dataset_id: str,
    recording_id: str,
    subject_id: str,
) -> None:
    registry.conn.execute(
        "INSERT OR IGNORE INTO recordings (recording_id) VALUES (?);",
        (recording_id,),
    )
    registry.conn.execute(
        """
        INSERT INTO recording_subjects (
            recording_id,
            subject_id,
            dataset_id,
            created_utc,
            updated_utc
        ) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(recording_id, subject_id) DO UPDATE SET
            dataset_id=excluded.dataset_id,
            updated_utc=excluded.updated_utc;
        """,
        (
            recording_id,
            subject_id,
            dataset_id,
            "2026-02-25T00:00:00+00:00",
            "2026-02-25T00:00:00+00:00",
        ),
    )
    registry.conn.commit()


def _write_manifest(path: Path, *, datasets: list[dict[str, object]]) -> None:
    payload = {
        "set_id": "eye_mask_smoke_v001",
        "set_version": 1,
        "source_type": "eye_masks",
        "query_filter": {"review_state": "approved", "review_intended_use": "training"},
        "datasets": datasets,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _profile_summary(
    *,
    heatmap_density: list[float],
    eye_sep: float,
    major: float,
    minor: float,
    left_area: float = 180.0,
    right_area: float = 185.0,
    union_area: float = 360.0,
    area_ratio: float = 0.97,
    left_area_usable: float | None = None,
    right_area_usable: float | None = None,
    union_area_usable: float | None = None,
) -> str:
    payload = {
        "quality": {
            "usable_rate": 0.9,
        },
        "geometry": {
            "eye_separation": {"stats": {"p50": eye_sep}},
            "ellipse_major": {"stats": {"p50": major}},
            "ellipse_minor": {"stats": {"p50": minor}},
            "left_area": {"stats": {"p50": left_area}},
            "right_area": {"stats": {"p50": right_area}},
            "union_area": {"stats": {"p50": union_area}},
            "area_lr_ratio": {"stats": {"p50": area_ratio}},
        },
        "spatial": {
            "edge_proximity_rate": 0.08,
            "center_heatmap": {
                "grid_h": 2,
                "grid_w": 2,
                "density": heatmap_density,
            },
        },
        "composition": {
            "rig_id": "rig_1",
            "arena_id": "arena_1",
            "dish_design": "cedar",
            "canvas_name": "shadow",
            "protocol_name": "DefaultScreen",
        },
    }
    if left_area_usable is not None:
        payload["geometry"]["left_area_usable"] = {"stats": {"p50": float(left_area_usable)}}
    if right_area_usable is not None:
        payload["geometry"]["right_area_usable"] = {"stats": {"p50": float(right_area_usable)}}
    if union_area_usable is not None:
        payload["geometry"]["union_area_usable"] = {"stats": {"p50": float(union_area_usable)}}
    return json.dumps(payload)


def test_aggregate_eye_mask_data_card_registry_first_payload_and_sections(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    manifest_path = tmp_path / "eye_mask.manifest.json"
    output_path = tmp_path / "eye_mask.data_card.json"

    zarr_a = tmp_path / "a_training.zarr"
    zarr_b = tmp_path / "b_training.zarr"

    db = Registry(registry_path)
    _seed_dataset(db, dataset_id="dataset_a", recording_id="rec_a", zarr_path=zarr_a)
    _seed_dataset(db, dataset_id="dataset_b", recording_id="rec_b", zarr_path=zarr_b)
    _seed_subject_lineage(db, dataset_id="dataset_a", recording_id="rec_a", subject_id="subject_a")
    _seed_subject_lineage(db, dataset_id="dataset_b", recording_id="rec_b", subject_id="subject_b")
    db.close()

    _write_manifest(
        manifest_path,
        datasets=[
            {"dataset_id": "dataset_a", "zarr_path": str(zarr_a)},
            {"dataset_id": "dataset_b", "zarr_path": str(zarr_b)},
        ],
    )

    fake_rows = {
        "dataset_a": {
            "dataset_id": "dataset_a",
            "profile_run": "eye_profile_a",
            "zarr_mtime_ns": int(zarr_a.stat().st_mtime_ns),
            "total_rois": 100,
            "successful_roi_pairs": 90,
            "successful_roi_pair_rate": 0.9,
            "rois_per_second": 4.0,
            "method": "refine_eye_masks",
            "camera_id": "cam_a",
            "rig_id": "rig_1",
            "arena_id": "arena_1",
            "dish_design": "cedar",
            "canvas_name": "shadow",
            "protocol_name": "DefaultScreen",
            "profile_json": _profile_summary(
                heatmap_density=[0.1, 0.2, 0.3, 0.4],
                eye_sep=5.2,
                major=8.0,
                minor=5.0,
                left_area=190.0,
                right_area=200.0,
                union_area=390.0,
                area_ratio=0.95,
            ),
            "review_state": "approved",
        },
        "dataset_b": {
            "dataset_id": "dataset_b",
            "profile_run": "eye_profile_b",
            "zarr_mtime_ns": int(zarr_b.stat().st_mtime_ns),
            "total_rois": 60,
            "successful_roi_pairs": 48,
            "successful_roi_pair_rate": 0.8,
            "rois_per_second": 3.0,
            "method": "refine_eye_masks",
            "camera_id": "cam_b",
            "rig_id": "rig_1",
            "arena_id": "arena_1",
            "dish_design": "cedar",
            "canvas_name": "shadow",
            "protocol_name": "DefaultScreen",
            "profile_json": _profile_summary(
                heatmap_density=[0.4, 0.3, 0.2, 0.1],
                eye_sep=5.4,
                major=8.3,
                minor=5.2,
                left_area=210.0,
                right_area=205.0,
                union_area=410.0,
                area_ratio=1.03,
            ),
            "review_state": "approved",
        },
    }

    def _fake_select_profile_rows(_registry: Registry, *, dataset_ids):
        return ({dataset_id: dict(fake_rows[dataset_id]) for dataset_id in dataset_ids}, "registry_sql_view")

    plot_calls: dict[str, object] = {}

    def _fake_generate_plots(*, card_payload, output_dir, prefix, heatmap_bin_factor):
        plot_calls["set_id"] = card_payload.get("set_id")
        plot_calls["output_dir"] = Path(output_dir)
        plot_calls["prefix"] = str(prefix)
        plot_calls["heatmap_bin_factor"] = int(heatmap_bin_factor)
        output_dir.mkdir(parents=True, exist_ok=True)
        fake_plot = output_dir / f"{prefix}.fake.png"
        fake_plot.write_bytes(b"PNG")
        return [fake_plot]

    monkeypatch.setattr(mod, "_select_profile_rows_registry_first", _fake_select_profile_rows)
    monkeypatch.setattr(mod.plot_data_card, "generate_eye_mask_training_data_card_plots", _fake_generate_plots)

    rc = mod.main(
        [
            "--manifest",
            str(manifest_path),
            "--registry",
            str(registry_path),
            "--output",
            str(output_path),
        ]
    )
    assert rc == 0
    assert output_path.exists()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    for key in (
        "selection",
        "quality",
        "geometry",
        "spatial",
        "composition",
        "subject_coverage",
        "parity",
        "audit_freshness",
    ):
        assert key in payload

    assert payload["schema_name"] == "eye_mask_training_data_card"
    assert payload["schema_version"] == "v1"
    assert payload["selection"]["dataset_count"] == 2
    assert payload["quality"]["total_rois"] == 160
    assert payload["quality"]["successful_roi_pairs_total"] == 138
    assert payload["quality"]["successful_roi_pair_rate_overall"] == pytest.approx(0.8625)
    assert payload["geometry"]["eye_separation_p50_dataset_stats"]["count"] == 2
    assert payload["geometry"]["left_area_p50_dataset_stats"]["p50"] == pytest.approx(200.0)
    assert payload["geometry"]["right_area_p50_dataset_stats"]["count"] == 2
    assert payload["geometry"]["union_area_p50_dataset_stats"]["mean"] == pytest.approx(400.0)
    assert payload["geometry"]["area_lr_ratio_p50_dataset_stats"]["count"] == 2
    assert payload["spatial"]["center_heatmap"]["grid_h"] == 2
    assert payload["composition"]["counts"]["camera_id"] == {"cam_a": 1, "cam_b": 1}
    assert payload["subject_coverage"]["lineage_covered_dataset_count"] == 2
    assert payload["audit_freshness"]["profile_source"] == "registry_sql_view"
    assert payload["audit_freshness"]["fallback_used"] is False
    assert len(payload["profile_run_refs"]) == 2
    source_refs = payload["audit_freshness"]["source_run_refs"]
    assert source_refs[0]["left_area_p50"] is not None
    assert source_refs[0]["right_area_p50"] is not None
    assert source_refs[0]["union_area_p50"] is not None

    assert plot_calls["set_id"] == "eye_mask_smoke_v001"
    assert plot_calls["prefix"] == "eye_mask_smoke_v001"
    assert plot_calls["output_dir"] == tmp_path / "eye_mask.data_card.plots"
    assert plot_calls["heatmap_bin_factor"] == 2


def test_aggregate_eye_mask_data_card_fallback_from_performance_latest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    manifest_path = tmp_path / "eye_mask.manifest.json"
    output_path = tmp_path / "eye_mask.data_card.json"

    zarr_a = tmp_path / "a_training.zarr"
    db = Registry(registry_path)
    _seed_dataset(db, dataset_id="dataset_a", recording_id="rec_a", zarr_path=zarr_a)
    _seed_subject_lineage(db, dataset_id="dataset_a", recording_id="rec_a", subject_id="subject_a")
    db.upsert_eye_mask_performance(
        dataset_id="dataset_a",
        stage_group="refined_eye_masks_runs",
        run_name="refined_eye_masks_001",
        run_created_utc="2026-02-25T00:00:00+00:00",
        recording_id="rec_a",
        zarr_use="training",
        method="refine_eye_masks",
        source_crop_run="crop_001",
        source_keypoint_group="keypoints_runs",
        source_keypoints_run="keypoints_001",
        source_eye_masks_run="eye_masks_001",
        source_eye_masks_method="traditional_eye_segmentation",
        total_rois=120,
        successful_eyes=220,
        successful_roi_pairs=102,
        successful_roi_pair_rate=0.85,
        duration_seconds=20.0,
        rois_per_second=6.0,
        inference_duration_seconds=None,
        inference_average_fps=6.0,
        reason_counts_json=json.dumps({"clean": 110, "manual_fix": 10}),
        summary_statistics_json=json.dumps(
            {
                "geometry": {
                    "eye_separation": {"stats": {"p50": 5.1}},
                    "ellipse_major": {"stats": {"p50": 8.2}},
                    "ellipse_minor": {"stats": {"p50": 5.3}},
                },
                "spatial": {
                    "edge_proximity_rate": 0.09,
                    "center_heatmap": {
                        "grid_h": 2,
                        "grid_w": 2,
                        "density": [0.25, 0.25, 0.25, 0.25],
                    },
                },
            }
        ),
        review_state="approved",
        review_method="manual",
        review_intended_use="training",
        review_reviewer="pytest",
        review_timestamp_utc="2026-02-25T00:05:00+00:00",
        zarr_mtime_ns=int(zarr_a.stat().st_mtime_ns),
    )
    db.close()

    _write_manifest(
        manifest_path,
        datasets=[
            {"dataset_id": "dataset_a", "zarr_path": str(zarr_a)},
        ],
    )

    def _fake_missing_profiles(_registry: Registry, *, dataset_ids):
        _ = dataset_ids
        return ({}, "missing_profile_view")

    monkeypatch.setattr(mod, "_select_profile_rows_registry_first", _fake_missing_profiles)

    rc_fail = mod.main(
        [
            "--manifest",
            str(manifest_path),
            "--registry",
            str(registry_path),
            "--output",
            str(output_path),
            "--no-plots",
        ]
    )
    assert rc_fail == 1
    assert not output_path.exists()

    rc_ok = mod.main(
        [
            "--manifest",
            str(manifest_path),
            "--registry",
            str(registry_path),
            "--output",
            str(output_path),
            "--allow-profile-fallback-scan",
            "--no-plots",
        ]
    )
    assert rc_ok == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["audit_freshness"]["fallback_used"] is True
    assert payload["audit_freshness"]["profile_source"] == "performance_latest_fallback"
    assert payload["quality"]["successful_roi_pair_rate_overall"] == pytest.approx(0.85)
    assert payload["geometry"]["eye_separation_p50_dataset_stats"]["p50"] == pytest.approx(5.1)


def test_query_eye_mask_performance_fallback_prefers_dataset_context_current(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        zarr_a = tmp_path / "a_training.zarr"
        _seed_dataset(registry, dataset_id="dataset_a", recording_id="rec_a", zarr_path=zarr_a)
        _seed_legacy_provenance_context(
            registry,
            dataset_id="dataset_a",
            rig_id="rig_legacy",
            arena_id="arena_legacy",
            camera_id="camera_legacy",
            canvas_name="canvas_legacy",
            protocol_name="protocol_legacy",
            dish_design="dish_design_legacy",
        )
        _seed_recording_context(
            registry,
            recording_id="rec_a",
            session_uuid="dataset_a_session",
            root=tmp_path,
            rig_id="rig_recording",
            arena_id="arena_recording",
            camera_id="camera_recording",
            canvas_name="canvas_recording",
            protocol_name="protocol_recording",
            dish_design="dish_design_recording",
        )
        registry.upsert_eye_mask_performance(
            dataset_id="dataset_a",
            stage_group="refined_eye_masks_runs",
            run_name="refined_eye_masks_001",
            run_created_utc="2026-02-25T00:00:00+00:00",
            recording_id="rec_a",
            zarr_use="training",
            method="refine_eye_masks",
            source_crop_run="crop_001",
            source_keypoint_group="keypoints_runs",
            source_keypoints_run="keypoints_001",
            source_eye_masks_run="eye_masks_001",
            source_eye_masks_method="traditional_eye_segmentation",
            total_rois=120,
            successful_eyes=220,
            successful_roi_pairs=102,
            successful_roi_pair_rate=0.85,
            duration_seconds=20.0,
            rois_per_second=6.0,
            inference_duration_seconds=None,
            inference_average_fps=6.0,
            reason_counts_json=json.dumps({"clean": 110}),
            summary_statistics_json=json.dumps({"geometry": {"eye_separation": {"stats": {"p50": 5.1}}}}),
            review_state="approved",
            review_method="manual",
            review_intended_use="training",
            review_reviewer="pytest",
            review_timestamp_utc="2026-02-25T00:05:00+00:00",
            zarr_mtime_ns=int(zarr_a.stat().st_mtime_ns),
        )

        rows = mod._query_eye_mask_performance_fallback(registry, dataset_ids=["dataset_a"])  # noqa: SLF001
    finally:
        registry.close()

    row = rows["dataset_a"]
    assert row["rig_id"] == "rig_recording"
    assert row["arena_id"] == "arena_recording"
    assert row["camera_id"] == "camera_recording"
    assert row["canvas_name"] == "canvas_recording"
    assert row["protocol_name"] == "protocol_recording"
    assert row["dish_design"] == "dish_design_recording"


def test_aggregate_eye_mask_data_card_profile_mtime_mismatch_policy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    manifest_path = tmp_path / "eye_mask.manifest.json"
    output_path = tmp_path / "eye_mask.data_card.json"

    zarr_a = tmp_path / "a_training.zarr"
    db = Registry(registry_path)
    _seed_dataset(db, dataset_id="dataset_a", recording_id="rec_a", zarr_path=zarr_a)
    db.close()

    _write_manifest(
        manifest_path,
        datasets=[
            {"dataset_id": "dataset_a", "zarr_path": str(zarr_a)},
        ],
    )

    row = {
        "dataset_id": "dataset_a",
        "profile_run": "eye_profile_a",
        "zarr_mtime_ns": int(zarr_a.stat().st_mtime_ns) + 10,
        "total_rois": 100,
        "successful_roi_pairs": 90,
        "successful_roi_pair_rate": 0.9,
        "rois_per_second": 4.0,
        "method": "refine_eye_masks",
        "profile_json": _profile_summary(
            heatmap_density=[0.1, 0.2, 0.3, 0.4],
            eye_sep=5.0,
            major=8.0,
            minor=5.0,
        ),
    }

    def _fake_select_profile_rows(_registry: Registry, *, dataset_ids):
        assert list(dataset_ids) == ["dataset_a"]
        return ({"dataset_a": dict(row)}, "registry_sql_view")

    monkeypatch.setattr(mod, "_select_profile_rows_registry_first", _fake_select_profile_rows)

    rc_fail = mod.main(
        [
            "--manifest",
            str(manifest_path),
            "--registry",
            str(registry_path),
            "--output",
            str(output_path),
            "--no-plots",
        ]
    )
    assert rc_fail == 1
    assert not output_path.exists()

    rc_ok = mod.main(
        [
            "--manifest",
            str(manifest_path),
            "--registry",
            str(registry_path),
            "--output",
            str(output_path),
            "--allow-profile-mtime-mismatch",
            "--no-plots",
        ]
    )
    assert rc_ok == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["audit_freshness"]["zarr_mtime_mismatch_count"] == 1


def test_aggregate_eye_mask_data_card_reports_low_area_sources(
    tmp_path: Path,
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    manifest_path = tmp_path / "eye_mask.manifest.json"
    output_path = tmp_path / "eye_mask.data_card.json"

    zarr_a = tmp_path / "a_training.zarr"
    zarr_b = tmp_path / "b_training.zarr"
    db = Registry(registry_path)
    _seed_dataset(db, dataset_id="dataset_a", recording_id="rec_a", zarr_path=zarr_a)
    _seed_dataset(db, dataset_id="dataset_b", recording_id="rec_b", zarr_path=zarr_b)
    db.close()

    _write_manifest(
        manifest_path,
        datasets=[
            {"dataset_id": "dataset_a", "zarr_path": str(zarr_a)},
            {"dataset_id": "dataset_b", "zarr_path": str(zarr_b)},
        ],
    )

    row_a = {
        "dataset_id": "dataset_a",
        "profile_run": "eye_profile_a",
        "zarr_mtime_ns": int(zarr_a.stat().st_mtime_ns),
        "total_rois": 100,
        "successful_roi_pairs": 90,
        "successful_roi_pair_rate": 0.9,
        "rois_per_second": 4.0,
        "method": "refine_eye_masks",
        "stage_group": "refined_eye_masks_runs",
        "review_state": "approved",
        "profile_json": _profile_summary(
            heatmap_density=[0.1, 0.2, 0.3, 0.4],
            eye_sep=5.1,
            major=8.0,
            minor=5.0,
            left_area=20.0,
            right_area=180.0,
            union_area=40.0,
            area_ratio=0.11,
        ),
    }
    row_b = {
        "dataset_id": "dataset_b",
        "profile_run": "eye_profile_b",
        "zarr_mtime_ns": int(zarr_b.stat().st_mtime_ns),
        "total_rois": 100,
        "successful_roi_pairs": 85,
        "successful_roi_pair_rate": 0.85,
        "rois_per_second": 4.0,
        "method": "refine_eye_masks",
        "stage_group": "refined_eye_masks_runs",
        "review_state": "approved",
        "profile_json": _profile_summary(
            heatmap_density=[0.4, 0.3, 0.2, 0.1],
            eye_sep=5.3,
            major=8.2,
            minor=5.1,
            left_area=180.0,
            right_area=190.0,
            union_area=360.0,
            area_ratio=0.95,
        ),
    }

    def _fake_select_profile_rows(_registry: Registry, *, dataset_ids):
        assert list(dataset_ids) == ["dataset_a", "dataset_b"]
        return (
            {
                "dataset_a": dict(row_a),
                "dataset_b": dict(row_b),
            },
            "registry_sql_view",
        )

    monkeypatch.setattr(mod, "_select_profile_rows_registry_first", _fake_select_profile_rows)

    rc = mod.main(
        [
            "--manifest",
            str(manifest_path),
            "--registry",
            str(registry_path),
            "--output",
            str(output_path),
            "--no-plots",
            "--report-min-eye-area-p50",
            "50",
            "--report-min-union-area-p50",
            "50",
        ]
    )
    assert rc == 0
    stdout = capsys.readouterr().out
    assert "Eye-mask low-area report:" in stdout
    assert "datasets_flagged=1/2" in stdout
    assert "dataset_a" in stdout
    assert "eye_area_p50_below_threshold" in stdout
    assert "union_area_p50_below_threshold" in stdout
    low_area_lines = [line for line in stdout.splitlines() if line.startswith("low_area\t")]
    assert len(low_area_lines) == 1
    assert "dataset_a" in low_area_lines[0]


def test_aggregate_eye_mask_data_card_prefers_usable_area_metrics_for_low_area_report(
    tmp_path: Path,
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    manifest_path = tmp_path / "eye_mask.manifest.json"
    output_path = tmp_path / "eye_mask.data_card.json"

    zarr_a = tmp_path / "a_training.zarr"
    db = Registry(registry_path)
    _seed_dataset(db, dataset_id="dataset_a", recording_id="rec_a", zarr_path=zarr_a)
    db.close()

    _write_manifest(
        manifest_path,
        datasets=[{"dataset_id": "dataset_a", "zarr_path": str(zarr_a)}],
    )

    row_a = {
        "dataset_id": "dataset_a",
        "profile_run": "eye_profile_a",
        "zarr_mtime_ns": int(zarr_a.stat().st_mtime_ns),
        "total_rois": 100,
        "successful_roi_pairs": 90,
        "successful_roi_pair_rate": 0.9,
        "rois_per_second": 4.0,
        "method": "refine_eye_masks",
        "stage_group": "refined_eye_masks_runs",
        "review_state": "approved",
        # Legacy registry columns can still carry all-row medians with many intentional empties.
        "left_area_p50": 0.0,
        "right_area_p50": 0.0,
        "union_area_p50": 0.0,
        "profile_json": _profile_summary(
            heatmap_density=[0.1, 0.2, 0.3, 0.4],
            eye_sep=5.1,
            major=8.0,
            minor=5.0,
            left_area=0.0,
            right_area=0.0,
            union_area=0.0,
            left_area_usable=180.0,
            right_area_usable=185.0,
            union_area_usable=360.0,
            area_ratio=0.11,
        ),
    }

    def _fake_select_profile_rows(_registry: Registry, *, dataset_ids):
        assert list(dataset_ids) == ["dataset_a"]
        return ({"dataset_a": dict(row_a)}, "registry_sql_view")

    monkeypatch.setattr(mod, "_select_profile_rows_registry_first", _fake_select_profile_rows)

    rc = mod.main(
        [
            "--manifest",
            str(manifest_path),
            "--registry",
            str(registry_path),
            "--output",
            str(output_path),
            "--no-plots",
            "--report-min-eye-area-p50",
            "50",
            "--report-min-union-area-p50",
            "50",
        ]
    )
    assert rc == 0
    stdout = capsys.readouterr().out
    assert "datasets_flagged=0/1" in stdout
    assert "low_area\tdataset_a" not in stdout

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    source_refs = payload["audit_freshness"]["source_run_refs"]
    assert len(source_refs) == 1
    expected_ratio = 180.0 / 185.0
    assert source_refs[0]["left_area_p50"] == pytest.approx(180.0)
    assert source_refs[0]["left_area_p50_all"] == pytest.approx(0.0)
    assert source_refs[0]["area_lr_ratio_p50"] == pytest.approx(expected_ratio)
    assert source_refs[0]["area_lr_ratio_p50_derived"] == pytest.approx(expected_ratio)
    assert source_refs[0]["area_lr_ratio_p50_profile"] == pytest.approx(0.11)
    assert source_refs[0]["area_lr_ratio_metric_source"] == "derived_from_selected_area_p50"
    assert source_refs[0]["area_metric_source"] == "usable"
    assert payload["geometry"]["area_lr_ratio_p50_dataset_stats"]["p50"] == pytest.approx(expected_ratio)


def test_aggregate_eye_mask_data_card_view_cannot_combine_dry_run() -> None:
    with pytest.raises(SystemExit) as exc:
        mod.main(
            [
                "--manifest",
                "/tmp/eye_mask.manifest.json",
                "--view",
                "--dry-run",
            ]
        )
    assert int(exc.value.code) == 2


def test_aggregate_eye_mask_data_card_view_and_force_use_plot_cli(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    manifest_path = tmp_path / "eye_mask.manifest.json"
    output_path = tmp_path / "eye_mask.data_card.json"

    zarr_a = tmp_path / "a_training.zarr"
    db = Registry(registry_path)
    _seed_dataset(db, dataset_id="dataset_a", recording_id="rec_a", zarr_path=zarr_a)
    db.close()

    _write_manifest(
        manifest_path,
        datasets=[
            {"dataset_id": "dataset_a", "zarr_path": str(zarr_a)},
        ],
    )

    row = {
        "dataset_id": "dataset_a",
        "profile_run": "eye_profile_a",
        "zarr_mtime_ns": int(zarr_a.stat().st_mtime_ns),
        "total_rois": 100,
        "successful_roi_pairs": 90,
        "successful_roi_pair_rate": 0.9,
        "rois_per_second": 4.0,
        "method": "refine_eye_masks",
        "profile_json": _profile_summary(
            heatmap_density=[0.1, 0.2, 0.3, 0.4],
            eye_sep=5.0,
            major=8.0,
            minor=5.0,
        ),
    }

    def _fake_select_profile_rows(_registry: Registry, *, dataset_ids):
        assert list(dataset_ids) == ["dataset_a"]
        return ({"dataset_a": dict(row)}, "registry_sql_view")

    plot_calls: dict[str, list[str]] = {}

    def _fake_plot_main(argv: list[str]) -> int:
        plot_calls["argv"] = list(argv)
        return 0

    def _unexpected_generate(*args, **kwargs):
        raise AssertionError("direct generator path should not be used when --view/--force are set")

    monkeypatch.setattr(mod, "_select_profile_rows_registry_first", _fake_select_profile_rows)
    monkeypatch.setattr(mod.plot_data_card, "main", _fake_plot_main)
    monkeypatch.setattr(mod.plot_data_card, "generate_eye_mask_training_data_card_plots", _unexpected_generate)

    rc = mod.main(
        [
            "--manifest",
            str(manifest_path),
            "--registry",
            str(registry_path),
            "--output",
            str(output_path),
            "--view",
            "--force",
        ]
    )
    assert rc == 0
    assert output_path.exists()
    plot_argv = plot_calls["argv"]
    assert "--card" in plot_argv
    assert str(output_path) in plot_argv
    assert "--view" in plot_argv
    assert "--force" in plot_argv
