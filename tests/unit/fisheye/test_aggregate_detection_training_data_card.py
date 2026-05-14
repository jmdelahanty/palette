from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.utils import aggregate_detection_training_data_card as mod


def test_refined_manifest_source_matches_instances_profile_surface() -> None:
    assert mod._detection_source_type_matches_manifest(
        manifest_source_type="refined",
        profile_detection_type="instances",
    )
    assert not mod._detection_source_type_matches_manifest(
        manifest_source_type="manual",
        profile_detection_type="instances",
    )


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
            "2026-02-24T00:00:00+00:00",
            "2026-02-24T00:00:00+00:00",
        ),
    )
    registry.conn.commit()


def _profile_json(*, camera_id: str, heatmap_density: list[float]) -> str:
    payload = {
        "schema_name": "detection_dataset_profile",
        "schema_version": "v1",
        "geometry_norm": {
            "w": {"count": 100},
            "h": {"count": 100},
            "area": {"count": 100},
            "aspect_ratio": {"count": 100},
        },
        "spatial": {
            "edge_proximity_rate": 0.05,
            "center_heatmap": {
                "grid_h": 2,
                "grid_w": 2,
                "density": heatmap_density,
            },
        },
        "histograms": {
            "w_norm": {"bin_edges": [0.0, 0.5, 1.0], "counts": [10, 90]},
            "h_norm": {"bin_edges": [0.0, 0.5, 1.0], "counts": [12, 88]},
            "area_norm": {"bin_edges": [0.0, 0.1, 1.0], "counts": [30, 70]},
            "aspect_ratio": {"bin_edges": [0.0, 1.0, 4.0], "counts": [40, 60]},
        },
        "composition": {
            "rig_id": "omnifin0",
            "camera_id": camera_id,
            "arena_id": "arena_1",
            "dish_design": "cedar",
            "canvas_name": "shadow",
            "protocol_name": "DefaultScreen",
        },
    }
    return json.dumps(payload)


def _upsert_profile(
    registry: Registry,
    *,
    dataset_id: str,
    profile_run: str,
    recording_id: str,
    zarr_path: Path,
    detection_type: str = "manual",
    detections_total: int = 100,
    frames_total: int = 100,
    frames_with_detections: int = 95,
    coverage_percent: float = 95.0,
    camera_id: str = "cam_a",
    heatmap_density: list[float] | None = None,
    genotype: str | None = None,
    dpf_at_acquisition: int | None = None,
) -> None:
    if heatmap_density is None:
        heatmap_density = [0.1, 0.2, 0.3, 0.4]
    registry.upsert_detection_data_profile(
        dataset_id=dataset_id,
        profile_run=profile_run,
        recording_id=recording_id,
        zarr_use="training",
        detection_type=detection_type,
        detection_path=f"refined_detect_runs/{profile_run}/manual",
        profile_created_utc="2026-02-24T00:00:00+00:00",
        zarr_mtime_ns=int(zarr_path.stat().st_mtime_ns),
        frames_total=frames_total,
        frames_with_detections=frames_with_detections,
        coverage_percent=coverage_percent,
        detections_total=detections_total,
        detections_per_frame_p50=1.0,
        detections_per_frame_p90=2.0,
        w_p10=0.02,
        w_p50=0.03,
        w_p90=0.04,
        h_p10=0.02,
        h_p50=0.03,
        h_p90=0.04,
        area_p10=0.001,
        area_p50=0.002,
        area_p90=0.003,
        aspect_ratio_p10=0.8,
        aspect_ratio_p50=1.0,
        aspect_ratio_p90=1.2,
        edge_proximity_rate=0.05,
        rig_id="omnifin0",
        camera_id=camera_id,
        arena_id="arena_1",
        dish_design="cedar",
        canvas_name="shadow",
        protocol_name="DefaultScreen",
        genotype=genotype,
        dpf_at_acquisition=dpf_at_acquisition,
        profile_json=_profile_json(camera_id=camera_id, heatmap_density=heatmap_density),
    )


def _write_manifest(path: Path, *, datasets: list[dict[str, object]]) -> None:
    payload = {
        "set_id": "detect_smoke_v001",
        "set_version": 1,
        "source_type": "manual",
        "query_filter": {"review_state": "approved", "review_intended_use": "training"},
        "datasets": datasets,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_aggregate_detection_training_data_card_writes_expected_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    manifest_path = tmp_path / "detect.manifest.json"
    output_path = tmp_path / "detect.data_card.json"

    zarr_a = tmp_path / "a_training.zarr"
    zarr_b = tmp_path / "b_training.zarr"
    db = Registry(registry_path)
    _seed_dataset(db, dataset_id="dataset_a", recording_id="rec_a", zarr_path=zarr_a)
    _seed_dataset(db, dataset_id="dataset_b", recording_id="rec_b", zarr_path=zarr_b)
    _upsert_profile(
        db,
        dataset_id="dataset_a",
        profile_run="profile_a",
        recording_id="rec_a",
        zarr_path=zarr_a,
        camera_id="cam_a",
        genotype="Tg(line_a)",
        dpf_at_acquisition=7,
    )
    _upsert_profile(
        db,
        dataset_id="dataset_b",
        profile_run="profile_b",
        recording_id="rec_b",
        zarr_path=zarr_b,
        camera_id="cam_b",
        detections_total=120,
        frames_total=110,
        frames_with_detections=110,
        coverage_percent=100.0,
        heatmap_density=[0.4, 0.3, 0.2, 0.1],
        genotype="Tg(line_b)",
        dpf_at_acquisition=8,
    )
    _seed_subject_lineage(db, dataset_id="dataset_a", recording_id="rec_a", subject_id="subject_a")
    _seed_subject_lineage(db, dataset_id="dataset_b", recording_id="rec_b", subject_id="subject_b")
    db.close()

    _write_manifest(
        manifest_path,
        datasets=[
            {
                "name": "dataset_a",
                "dataset_id": "dataset_a",
                "zarr_path": str(zarr_a),
                "detection_source_type": "manual",
            },
            {
                "name": "dataset_b",
                "dataset_id": "dataset_b",
                "zarr_path": str(zarr_b),
                "detection_source_type": "manual",
            },
        ],
    )

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

    monkeypatch.setattr(mod.plot_data_card, "generate_detection_training_data_card_plots", _fake_generate_plots)

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
    assert payload["schema_name"] == "detection_training_data_card"
    assert payload["schema_version"] == "v1"
    assert payload["set_id"] == "detect_smoke_v001"
    assert payload["selection"]["dataset_count"] == 2
    assert payload["coverage"]["frames_total"] == 210
    assert payload["coverage"]["frames_with_detections"] == 205
    assert payload["counts"]["detections_total"] == 220
    assert payload["composition_counts"]["camera_id"] == {"cam_a": 1, "cam_b": 1}
    assert payload["subject_coverage"] == {
        "manifest_dataset_count": 2,
        "lineage_covered_dataset_count": 2,
        "missing_lineage_dataset_ids": [],
    }
    assert payload["genotype_counts"] == {"Tg(line_a)": 1, "Tg(line_b)": 1}
    assert payload["dpf_stats"]["count"] == 2
    assert payload["dpf_stats"]["min"] == 7.0
    assert payload["dpf_stats"]["max"] == 8.0
    assert payload["dpf_stats"]["mean"] == 7.5
    assert payload["dpf_histogram"]["bin_edges"] == [6.5, 7.5, 8.5]
    assert payload["dpf_histogram"]["counts"] == [1, 1]
    assert payload["dpf_histogram"]["source_dataset_count"] == 2
    assert payload["dpf_histogram"]["skipped_missing_values"] == 0
    assert len(payload["profile_run_refs"]) == 2
    assert payload["histograms_aggregate"]["w_norm"]["counts"] == [20, 180]
    center = payload["spatial_aggregate"]["center_heatmap"]
    assert center["grid_h"] == 2
    assert center["grid_w"] == 2
    assert len(center["density"]) == 4
    assert pytest.approx(sum(center["density"]), rel=1e-6) == 1.0
    assert plot_calls["set_id"] == "detect_smoke_v001"
    assert plot_calls["prefix"] == "detect_smoke_v001"
    assert plot_calls["output_dir"] == tmp_path / "detect.data_card.plots"
    assert plot_calls["heatmap_bin_factor"] == 2
    assert (tmp_path / "detect.data_card.plots" / "detect_smoke_v001.fake.png").exists()


def test_aggregate_detection_training_data_card_fails_when_profile_missing(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    manifest_path = tmp_path / "detect.manifest.json"
    output_path = tmp_path / "detect.data_card.json"

    zarr_a = tmp_path / "a_training.zarr"
    zarr_b = tmp_path / "b_training.zarr"
    db = Registry(registry_path)
    _seed_dataset(db, dataset_id="dataset_a", recording_id="rec_a", zarr_path=zarr_a)
    _seed_dataset(db, dataset_id="dataset_b", recording_id="rec_b", zarr_path=zarr_b)
    _upsert_profile(db, dataset_id="dataset_a", profile_run="profile_a", recording_id="rec_a", zarr_path=zarr_a)
    db.close()

    _write_manifest(
        manifest_path,
        datasets=[
            {
                "name": "dataset_a",
                "dataset_id": "dataset_a",
                "zarr_path": str(zarr_a),
                "detection_source_type": "manual",
            },
            {
                "name": "dataset_b",
                "dataset_id": "dataset_b",
                "zarr_path": str(zarr_b),
                "detection_source_type": "manual",
            },
        ],
    )

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
    assert rc == 1
    assert not output_path.exists()


def test_aggregate_detection_data_card_prefers_registry_dataset_id_for_manifest_rows(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    manifest_path = tmp_path / "detect.manifest.json"
    output_path = tmp_path / "detect.data_card.json"

    zarr_path = tmp_path / "legacy_training.zarr"
    db = Registry(registry_path)
    _seed_dataset(
        db,
        dataset_id="rec_legacy:z123456789abc",
        recording_id="rec_legacy",
        zarr_path=zarr_path,
    )
    _upsert_profile(
        db,
        dataset_id="rec_legacy:z123456789abc",
        profile_run="profile_legacy",
        recording_id="rec_legacy",
        zarr_path=zarr_path,
    )
    db.close()

    _write_manifest(
        manifest_path,
        datasets=[
            {
                "name": "legacy_dataset",
                # Intentional legacy-style manifest id (recording/session id).
                "dataset_id": "rec_legacy",
                "zarr_path": str(zarr_path),
                "detection_source_type": "manual",
            }
        ],
    )

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
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["profile_run_refs"][0]["dataset_id"] == "rec_legacy:z123456789abc"


def test_aggregate_detection_training_data_card_no_plots_flag_disables_plot_generation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    manifest_path = tmp_path / "detect.manifest.json"
    output_path = tmp_path / "detect.data_card.json"

    zarr_a = tmp_path / "a_training.zarr"
    db = Registry(registry_path)
    _seed_dataset(db, dataset_id="dataset_a", recording_id="rec_a", zarr_path=zarr_a)
    _upsert_profile(db, dataset_id="dataset_a", profile_run="profile_a", recording_id="rec_a", zarr_path=zarr_a)
    db.close()

    _write_manifest(
        manifest_path,
        datasets=[
            {
                "name": "dataset_a",
                "dataset_id": "dataset_a",
                "zarr_path": str(zarr_a),
                "detection_source_type": "manual",
            }
        ],
    )

    called = {"plots": False}

    def _fake_generate_plots(*, card_payload, output_dir, prefix, heatmap_bin_factor):
        called["plots"] = True
        return []

    monkeypatch.setattr(mod.plot_data_card, "generate_detection_training_data_card_plots", _fake_generate_plots)

    rc = mod.main(
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
    assert rc == 0
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["subject_coverage"] == {
        "manifest_dataset_count": 1,
        "lineage_covered_dataset_count": 0,
        "missing_lineage_dataset_ids": ["dataset_a"],
    }
    assert payload["genotype_counts"] == {}
    assert payload["dpf_stats"] is None
    assert payload["dpf_histogram"] is None
    assert called["plots"] is False
    assert not (tmp_path / "detect.data_card.plots").exists()


def test_aggregate_detection_training_data_card_mtime_validation(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    manifest_path = tmp_path / "detect.manifest.json"
    output_path = tmp_path / "detect.data_card.json"

    zarr_a = tmp_path / "a_training.zarr"
    db = Registry(registry_path)
    _seed_dataset(db, dataset_id="dataset_a", recording_id="rec_a", zarr_path=zarr_a)
    _upsert_profile(db, dataset_id="dataset_a", profile_run="profile_a", recording_id="rec_a", zarr_path=zarr_a)
    db.conn.execute(
        "UPDATE detection_data_profile SET zarr_mtime_ns = zarr_mtime_ns + 12345 WHERE dataset_id = ?;",
        ("dataset_a",),
    )
    db.conn.commit()
    db.close()

    _write_manifest(
        manifest_path,
        datasets=[
            {
                "name": "dataset_a",
                "dataset_id": "dataset_a",
                "zarr_path": str(zarr_a),
                "detection_source_type": "manual",
            }
        ],
    )

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
    assert rc == 1
    assert not output_path.exists()

    rc = mod.main(
        [
            "--manifest",
            str(manifest_path),
            "--registry",
            str(registry_path),
            "--output",
            str(output_path),
            "--allow-mtime-mismatch",
        ]
    )
    assert rc == 0
    assert output_path.exists()


def test_aggregate_detection_training_data_card_subject_lineage_warn_policy(
    tmp_path: Path,
    capsys,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    manifest_path = tmp_path / "detect.manifest.json"
    output_path = tmp_path / "detect.data_card.json"

    zarr_a = tmp_path / "a_training.zarr"
    zarr_b = tmp_path / "b_training.zarr"
    db = Registry(registry_path)
    _seed_dataset(db, dataset_id="dataset_a", recording_id="rec_a", zarr_path=zarr_a)
    _seed_dataset(db, dataset_id="dataset_b", recording_id="rec_b", zarr_path=zarr_b)
    _upsert_profile(
        db,
        dataset_id="dataset_a",
        profile_run="profile_a",
        recording_id="rec_a",
        zarr_path=zarr_a,
        genotype="Tg(line_warn)",
        dpf_at_acquisition=7,
    )
    _upsert_profile(db, dataset_id="dataset_b", profile_run="profile_b", recording_id="rec_b", zarr_path=zarr_b)
    _seed_subject_lineage(db, dataset_id="dataset_a", recording_id="rec_a", subject_id="subject_a")
    db.close()

    _write_manifest(
        manifest_path,
        datasets=[
            {
                "name": "dataset_a",
                "dataset_id": "dataset_a",
                "zarr_path": str(zarr_a),
                "detection_source_type": "manual",
            },
            {
                "name": "dataset_b",
                "dataset_id": "dataset_b",
                "zarr_path": str(zarr_b),
                "detection_source_type": "manual",
            },
        ],
    )

    rc = mod.main(
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
    assert rc == 0
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["subject_coverage"] == {
        "manifest_dataset_count": 2,
        "lineage_covered_dataset_count": 1,
        "missing_lineage_dataset_ids": ["dataset_b"],
    }
    assert payload["genotype_counts"] == {"Tg(line_warn)": 1}
    assert payload["dpf_stats"]["count"] == 1
    assert payload["dpf_stats"]["min"] == 7.0
    assert payload["dpf_stats"]["max"] == 7.0
    assert payload["dpf_histogram"]["bin_edges"] == [6.5, 7.5]
    assert payload["dpf_histogram"]["counts"] == [1]
    assert payload["dpf_histogram"]["source_dataset_count"] == 1
    assert payload["dpf_histogram"]["skipped_missing_values"] == 1
    stdout = capsys.readouterr().out
    assert "Subject lineage coverage: 1/2 datasets" in stdout
    assert "Subject lineage missing dataset_id(s): dataset_b" in stdout


def test_aggregate_detection_training_data_card_subject_lineage_require_policy(
    tmp_path: Path,
    capsys,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    manifest_path = tmp_path / "detect.manifest.json"
    output_path = tmp_path / "detect.data_card.json"

    zarr_a = tmp_path / "a_training.zarr"
    zarr_b = tmp_path / "b_training.zarr"
    db = Registry(registry_path)
    _seed_dataset(db, dataset_id="dataset_a", recording_id="rec_a", zarr_path=zarr_a)
    _seed_dataset(db, dataset_id="dataset_b", recording_id="rec_b", zarr_path=zarr_b)
    _upsert_profile(db, dataset_id="dataset_a", profile_run="profile_a", recording_id="rec_a", zarr_path=zarr_a)
    _upsert_profile(db, dataset_id="dataset_b", profile_run="profile_b", recording_id="rec_b", zarr_path=zarr_b)
    _seed_subject_lineage(db, dataset_id="dataset_a", recording_id="rec_a", subject_id="subject_a")
    db.close()

    _write_manifest(
        manifest_path,
        datasets=[
            {
                "name": "dataset_a",
                "dataset_id": "dataset_a",
                "zarr_path": str(zarr_a),
                "detection_source_type": "manual",
            },
            {
                "name": "dataset_b",
                "dataset_id": "dataset_b",
                "zarr_path": str(zarr_b),
                "detection_source_type": "manual",
            },
        ],
    )

    rc = mod.main(
        [
            "--manifest",
            str(manifest_path),
            "--registry",
            str(registry_path),
            "--output",
            str(output_path),
            "--subject-lineage-policy",
            "require",
            "--no-plots",
        ]
    )
    assert rc == 1
    assert not output_path.exists()
    stdout = capsys.readouterr().out
    assert (
        "Training data card aggregation failed: Missing subject lineage rows in "
        "recording_subject_overview for dataset_id(s): dataset_b"
    ) in stdout
